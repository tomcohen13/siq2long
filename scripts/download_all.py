# Adapted from the original SIQ codebase for SIQ2-Full.
# Two modes, selected with --mode:
#   siq2      - trim video and transcript to the 60s oracle window given by trims.json
#   siq2long - full-length video and transcript, nothing is trimmed
import argparse
import collections
import concurrent.futures
import contextlib
import datetime
import glob
import json
import os
import shutil
import subprocess
import tempfile
import threading

import webvtt
from tqdm import tqdm

import youtube_utils

join = os.path.join
TRIM_SECONDS = 60
SUBSETS = ('youtubeclips', 'movieclips', 'car')
SPLITS = ('train', 'val', 'test')

def parse_args():
    parser = argparse.ArgumentParser(description='Command-line arguments')
    parser.add_argument('--data_dir', type=str, default='siq2long',
                        help='Input directory holding trims.json and original_split.json')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where video/transcript/audio/frames are written. Defaults to a '
                             'sibling of --data_dir named after --mode.')
    parser.add_argument('--mode', type=str, default='siq2long', choices=['siq2long', 'siq2'],
                        help='siq2: trim video and transcript to the oracle window. '
                             'siq2long: keep them full length.')
    parser.add_argument('--splits', type=str, nargs='+', default=list(SPLITS), choices=SPLITS,
                        metavar='SPLIT',
                        help='Which splits to download, in the order given, e.g. --splits val '
                             'train. Splits left out keep whatever current_split.json already '
                             'records for them. Default: ' + ' '.join(SPLITS) + '.')
    parser.add_argument('--blacklist', type=str, default=None,
                        help='JSONL log of (id, artifact) pairs that failed permanently -- '
                             'private, removed, geo-blocked, captionless. Skipped on later runs, '
                             'so a retry pass spends its requests on the rate-limited ids instead. '
                             'Defaults to blacklist.jsonl inside the output dir; point both modes '
                             'at one path to share it. Delete a line to retry that artifact.')
    parser.add_argument('--workers', type=int, default=3,
                        help='Parallel downloads. Network-bound, so this is the main speed knob, '
                             'but YouTube answers unpaced parallel requests with HTTP 429.')
    parser.add_argument('--cookies_from_browser', type=str, default=None,
                        help="Pull YouTube cookies from a browser: 'chrome', or 'chrome:Profile 2' "
                             "to pick a non-default profile. Needed for age-gated videos.")
    parser.add_argument('--cookies_file', type=str, default=None,
                        help='Path to an exported cookies.txt. Takes precedence over '
                             '--cookies_from_browser and touches no browser files.')
    parser.add_argument('--frames', action='store_true',
                        help='Also extract 3fps JPEG frames. Off by default: frame sampling is '
                             'per-model and expensive on full-length video.')
    parser.add_argument('--audio', action='store_true',
                        help='Also extract an audio-only mp3. Off by default: the VLM baselines '
                             'do not ingest audio.')
    return parser.parse_args()

@contextlib.contextmanager
def atomic(final_path):
    """Yield a sibling '.part' path and move it into place only on success.

    Without this, a killed ffmpeg leaves a truncated file at the final path, where the
    need_* checks would accept it as complete. Adds no truncation detection: an existing
    complete file is never examined, so it is never re-downloaded.
    """
    tmp = final_path + '.part'
    try:
        yield tmp
        os.replace(tmp, final_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def load_blacklist(path):
    """Read the (id, artifact) -> reason pairs recorded as permanently failed."""
    entries = {}
    if not os.path.exists(path):
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue  # a run killed mid-append leaves one torn line; the rest is good
            entries[(record['id'], record['artifact'])] = record['reason']
    return entries

def blacklist_add(id, artifact, reason, detail=''):
    """Record a permanent failure.

    Appends one line rather than rewriting, so a killed run keeps every entry already
    written and concurrent workers cannot clobber each other's.
    """
    if (id, artifact) in blacklist:
        return
    blacklist[(id, artifact)] = reason
    record = {'id': id, 'artifact': artifact, 'reason': reason,
              'when': datetime.datetime.now().isoformat(timespec='seconds')}
    if detail:
        record['error'] = detail[:500]
    with blacklist_lock:
        with open(blacklist_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
    tqdm.write('blacklisted {id} ({artifact}): {reason}'.format(
        id=id, artifact=artifact, reason=reason))

def english_caption_tracks(info):
    """English tracks the info json advertises, manual or automatic."""
    tracks = dict(info.get('subtitles') or {})
    tracks.update(info.get('automatic_captions') or {})
    return [lang for lang in tracks if lang.lower().startswith('en')]

def vtt_timestamp(delta):
    """Format a timedelta as a WebVTT HH:MM:SS.mmm timestamp."""
    ms = int(round(delta.total_seconds() * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return '%02d:%02d:%02d.%03d' % (h, m, s, ms)

def trim_transcript(transcript, trim_time):
    """Keep only captions inside the TRIM_SECONDS window at trim_time, rebased to start at 0.

    trims.json stores plain float seconds. The original read the digits after the decimal
    point as milliseconds, which only holds at 3 digits ("59.98" became 59.098).
    """
    epoch = datetime.datetime(2000, 1, 1)
    window_start = epoch + datetime.timedelta(seconds=trim_time)
    window_end = window_start + datetime.timedelta(seconds=TRIM_SECONDS)

    trimmed = webvtt.WebVTT()
    for caption in transcript:
        start = datetime.datetime.strptime(caption.start, '%H:%M:%S.%f').replace(year=2000, month=1, day=1)
        end = datetime.datetime.strptime(caption.end, '%H:%M:%S.%f').replace(year=2000, month=1, day=1)
        if start >= window_start and end <= window_end:
            caption.start = vtt_timestamp(start - window_start)
            caption.end = vtt_timestamp(end - window_start)
            trimmed.captions.append(caption)
    return trimmed

def process_video(id):
    """Download and write every artifact for one id.

    Runs on a worker thread and touches no shared state; the caller aggregates the status,
    one of 'ok', 'no_trim', 'no_video', 'no_transcript', or an 'error: ...' string.
    """
    if id not in trims:
        return 'no_trim'

    # Skip per artifact, not per id: a previous run could write the mp4 and still fail on
    # the transcript, and skipping the whole id left those permanently incomplete.
    video_file = join(video_path, id + '.mp4')
    mp3_file = join(mp3_path, id + '.mp3')
    vtt_file = join(transcript_path, id + '.vtt')
    # only holds artifacts this run actually expects, so the skip message below cannot
    # claim an audio/frames file that was never requested
    need = {
        'video': not os.path.exists(video_file),
        'transcript': not os.path.exists(vtt_file),
    }
    if args.audio:
        need['audio'] = not os.path.exists(mp3_file)
    if args.frames:
        need['frames'] = not os.path.isdir(join(frame_dir, id))
    if not any(need.values()):
        # tqdm.write rather than print, so the line does not tear the progress bar
        tqdm.write('skipped {id}: found all expected files ({have})'.format(
            id=id, have=', '.join(k for k, v in need.items() if not v)))
        return 'ok'

    # Retrying a permanent failure only spends a request to fail the same way, which is the
    # whole cost this blacklist exists to avoid. The id is still reported and still left out
    # of current_split.json, exactly as the original failure left it.
    if any(need.get(artifact) and (id, artifact) in blacklist
           for artifact in ('video', 'transcript')):
        return 'blacklisted'

    trim_time = float(trims[id])
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            if need['video']:
                try:
                    full_video = youtube_utils.download_video(id, temp_dir, False)
                except youtube_utils.PermanentFailure as e:
                    blacklist_add(id, 'video', e.reason, e.message)
                    return 'no_video'
                if full_video is None:
                    return 'no_video'
                # -ss before -i seeks the input rather than decoding up to the cut. In
                # siq2long the download is kept verbatim, so the copy needs no transcode --
                # download_video has already verified it is playable.
                with atomic(video_file) as tmp:
                    if trim:
                        subprocess.run(['ffmpeg', '-y', '-v', 'error',
                                        '-ss', str(trim_time), '-i', full_video,
                                        '-t', str(TRIM_SECONDS),
                                        '-c:v', 'libx264', '-c:a', 'aac',
                                        '-f', 'mp4', tmp], check=True)
                    else:
                        shutil.copy(full_video, tmp)

            # One extraction straight from the mp4; the original went mp4 -> mp3 -> wav, so
            # the wav carried the mp3's artifacts for no benefit.
            if need.get('audio'):
                with atomic(mp3_file) as tmp:
                    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', video_file,
                                    '-vn', '-ar', '22050', '-ac', '1', '-f', 'mp3', tmp],
                                   check=True)

            if need['transcript']:
                try:
                    _, info = youtube_utils.download_transcript(id, temp_dir)
                except youtube_utils.PermanentFailure as e:
                    blacklist_add(id, 'transcript', e.reason, e.message)
                    return 'no_transcript'
                # Track codes are no longer plain 'en' -- YouTube emits 'en-<trackid>' -- so
                # fall back to any English variant instead of two hardcoded suffixes.
                candidates = [join(temp_dir, id + '.v2.en.vtt'),
                              join(temp_dir, id + '.v2.en-manual.vtt')]
                candidates += sorted(glob.glob(join(temp_dir, id + '.v2.en*.vtt')))
                for path in candidates:
                    try:
                        transcript = webvtt.read(path)
                        break
                    except Exception:
                        continue
                else:
                    # yt-dlp raised nothing and still wrote no English track. If the info
                    # json advertises none either, the video simply has no captions and a
                    # later run would fetch the same nothing; if it does advertise one, the
                    # fetch itself failed, which is worth retrying.
                    if info and not english_caption_tracks(info):
                        blacklist_add(id, 'transcript', 'no_captions')
                    return 'no_transcript'

                out = trim_transcript(transcript, trim_time) if trim else transcript
                # write(f) rather than save(path): save() appends '.vtt' to any target not
                # already ending in it, which would mangle the '.part' name
                with atomic(vtt_file) as tmp:
                    with open(tmp, 'w', encoding='utf-8') as f:
                        out.write(f)

            if need.get('frames'):
                os.makedirs(join(frame_dir, id), exist_ok=True)
                subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', video_file,
                                '-r', '3', '-q:v', '1',
                                join(frame_dir, id, id + '_%03d.jpg')], check=True)
        return 'ok'
    except Exception as e:
        return 'error: {err}'.format(err=e)

def find_active_videos(ids, desc):
    """Process ids in parallel, then report in input order so the split stays reproducible."""
    statuses = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_video, id): id for id in ids}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(ids),
                           desc='Downloading ' + desc):
            statuses[futures[future]] = future.result()

    for id in ids:
        status = statuses.get(id, '')
        if status.startswith('error'):
            failures['errors'].append('{id}: {status}'.format(id=id, status=status))
        elif status != 'ok':
            failures[status].append(id)

    return [id for id in ids if statuses.get(id) == 'ok']

def dedupe_split(subsets):
    """Drop ids appearing in more than one split, keeping the most eval-ward one.

    The original SIQ2 split repeats a few ids across subsets, which leaks between train and
    eval. Keeping test > val > train means an eval video is never also a training video.
    Returns (id, dropped_from, kept_in) tuples for reporting.
    """
    priority = {'test': 0, 'val': 1, 'train': 2}
    owner = {}
    for _, subset, sp in sorted((priority[sp], subset, sp) for subset in subsets for sp in subsets[subset]):
        for id in subsets[subset][sp]:
            owner.setdefault(id, (subset, sp))

    dropped = []
    for subset in subsets:
        for sp in subsets[subset]:
            kept, seen = [], set()
            for id in subsets[subset][sp]:
                if owner[id] == (subset, sp) and id not in seen:
                    seen.add(id)
                    kept.append(id)
                else:
                    dropped.append((id, '%s/%s' % (subset, sp), '%s/%s' % owner[id]))
            subsets[subset][sp] = kept
    return dropped

def main():
    # process_video runs on worker threads and find_active_videos aggregates into failures,
    # so both read this state at module scope; main binds it there rather than locally. The
    # __main__ guard is the part that matters -- importing this file downloads nothing.
    global args, trim, trims, failures, blacklist, blacklist_lock, blacklist_path
    global transcript_path, video_path, mp3_path, frame_dir

    args = parse_args()
    trim = args.mode == 'siq2'
    youtube_utils.COOKIES_FROM_BROWSER = args.cookies_from_browser
    youtube_utils.COOKIES_FILE = args.cookies_file

    # The mode names the output directory, as a sibling of data_dir, so trimmed and full-length
    # artifacts can never share one -- the skip check cannot tell them apart.
    output_dir = args.output_dir or join(os.path.dirname(args.data_dir), args.mode)

    trims_path = join(args.data_dir, 'trims.json')
    original_split_path = join(args.data_dir, 'original_split.json')
    current_split_path = join(output_dir, 'current_split.json')
    transcript_path = join(output_dir, 'transcript')
    video_path = join(output_dir, 'video')
    mp3_path = join(output_dir, 'audio', 'mp3')
    frame_dir = join(output_dir, 'frames')

    # An existing output dir is a resumed run. A missing one is a new dataset on disk, so
    # confirm the destination before creating it.
    if not os.path.isdir(output_dir):
        print('Output directory does not exist yet:', os.path.abspath(output_dir))
        print("Mode '{mode}' reads metadata from {data}".format(mode=args.mode, data=os.path.abspath(args.data_dir)))
        if input('Create it and download {mode} artifacts there? [y/N] '.format(mode=args.mode)).strip().lower() not in ('y', 'yes'):
            raise SystemExit('Aborted, nothing downloaded.')
        print('Creating', os.path.abspath(output_dir))

    os.makedirs(transcript_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)
    if args.audio:
        os.makedirs(mp3_path, exist_ok=True)

    # siq2long keeps whole videos, so the footprint is several times the trimmed set
    print('free space at {dir}: {gb:.1f} GB'.format(
        dir=output_dir, gb=shutil.disk_usage(output_dir).free / 1e9))

    blacklist_path = args.blacklist or join(output_dir, 'blacklist.jsonl')
    # --blacklist may point outside the output dir, and failing to open it would surface hours
    # in, on a worker thread, with the run already underway
    os.makedirs(os.path.dirname(os.path.abspath(blacklist_path)), exist_ok=True)
    blacklist_lock = threading.Lock()
    blacklist = load_blacklist(blacklist_path)
    blacklist_at_start = len(blacklist)
    if blacklist:
        reasons = collections.Counter(blacklist.values())
        print('blacklist: {n} entries at {path} ({detail})'.format(
            n=blacklist_at_start, path=blacklist_path,
            detail=', '.join('%s %s' % (count, reason) for reason, count in reasons.most_common())))

    with open(trims_path) as f:
        trims = json.load(f)
    with open(original_split_path) as f:
        split = json.load(f)

    failures = {'no_trim': [], 'no_video': [], 'no_transcript': [], 'blacklisted': [], 'errors': []}

    duplicates = dedupe_split(split['subsets'])
    if duplicates:
        print('dropped %d duplicate id(s) from the split:' % len(duplicates))
        for id, dropped_from, kept_in in duplicates:
            print('  {id}: dropped from {a}, kept in {b}'.format(id=id, a=dropped_from, b=kept_in))

    # Seed from the manifest already on disk, so a run over a subset of splits leaves the rest
    # of it alone. Writing only the splits this run touched would drop every other split from
    # the dataset manifest.
    new_split = {'subsets': {subset: {sp: [] for sp in SPLITS} for subset in SUBSETS}}
    if os.path.exists(current_split_path):
        with open(current_split_path) as f:
            prior = json.load(f)
        for subset in SUBSETS:
            for sp in SPLITS:
                new_split['subsets'][subset][sp] = prior.get('subsets', {}).get(subset, {}).get(sp, [])

    # Splits outermost, so --splits val train finishes all of val before it starts train
    for sp in args.splits:
        for subset in SUBSETS:
            new_split['subsets'][subset][sp] = find_active_videos(
                split['subsets'][subset][sp], '%s/%s' % (subset, sp))
            # Persist after each pass. A full-length run takes hours, and writing only at the
            # end meant an interrupt discarded the manifest for everything already downloaded.
            with open(current_split_path, 'w') as f:
                json.dump(new_split, f)

    LABELS = {
        'no_trim': 'could not find trims for',
        'no_video': 'could not download videos for',
        'no_transcript': 'could not download transcripts for',
        'blacklisted': 'skipped as permanently failed',
        'errors': 'failed with errors',
    }
    for key, label in LABELS.items():
        if failures[key]:
            print('\n{label} ({n}):'.format(label=label, n=len(failures[key])))
            # the blacklist grows into the hundreds, and dumping all of it buries the rest
            for item in failures[key][:20]:
                print(' ', item)
            if len(failures[key]) > 20:
                print('  ... and {n} more, see {path}'.format(
                    n=len(failures[key]) - 20, path=blacklist_path if key == 'blacklisted' else 'above'))

    added = len(blacklist) - blacklist_at_start
    if added:
        print('\nadded {n} entries to {path}'.format(n=added, path=blacklist_path))

if __name__ == "__main__":
    main()
