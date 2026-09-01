"""
Simple library for reading youtube VTT files
author: Sheryl Mathew
"""
import glob
import json
import os
import re
import subprocess

# import googleapiclient.discovery
import numpy as np
from bs4 import BeautifulSoup
# from googleapiclient.http import HttpError
from yt_dlp import YoutubeDL, DownloadError
# from youtube_dl import YoutubeDL, DownloadError
# from youtube_dl.utils import subtitles_filename, ExtractorError, encodeFilename
from yt_dlp.utils import subtitles_filename, ExtractorError, encodeFilename
import io
import time

def ts_to_sec(ts):
    """
    Splits a timestamp of the form HH:MM:SS.MIL
    :param ts:  timestamp of the form HH:MM:SS.MIL
    :return: seconds
    """
    rest, ms = ts.split('.')
    hh, mm, ss = rest.split(':')
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + float('.{}'.format(ms))


def sec_to_ts(sec):
    """
    seconds to timestamp
    :param sec:  number of seconds
    :return: timestamp of the form HH:MM:SS.MIL
    """
    ms = '{:.3f}'.format(sec).split('.')[-1]
    int_time = int(sec)
    ss = int_time % 60
    int_time = int_time // 60
    mm = int_time % 60
    hh = int_time // 60
    return '{:0>2d}:{:0>2d}:{:0>2d}.{}'.format(hh, mm, ss, ms)


def _read_part(stuff, start_ts, stop_ts):
    """
    Reads a part of a VTT subtitles file

    My current understanding is that the important stuff looks like this

    00:00:00.030 --> 00:00:02.060 align:start position:0%

    hello<00:00:00.450><c> everyone</c><00:00:00.840><c> and</c><00:00:01.140><c> welcome</c><00:00:01.199><c> in</c><00:00:01.800><c> this</c><00:00:01.890><c> video</c>

    so Hello started at 00:00:00.030 and goes to 00:00:00.450, everyone started right after and goes to 00:00:00.840

    We will extract the start and stop timesteps

    :param stuff: VTT text between two timestamps.
    :param ts: Initial timestamp.
    :return:
     (word, start time, end time, start time of chunk, end time of chunk, distance from left, distance from right)
    """
    # print("Start: {} stop: {} \n stuff {} \n\n".format(start, stop, stuff), flush=True)
    matching_lines = re.findall(r'^(.+<\d\d:\d\d:\d\d\.\d\d\d>.+)$', '\n'.join(stuff), flags=re.MULTILINE)

    start_time = ts_to_sec(start_ts)
    end_time = ts_to_sec(stop_ts)

    if len(matching_lines) == 0:
        # EXCEPTION: IF there is a single word, that is on the second line, and we didn't add anything
        if len(stuff) >= 3 and len(stuff[1].strip()) > 0 and ('<c>' not in stuff[1]) and len(stuff[1].strip().split(' ')) > 0:
            return [(stuff[1].strip(), start_time, end_time)]
        else:
            return []
    if not len(matching_lines) == 1:
        raise ValueError("WTF? VTT subtitles not well formed:\n{}".format('\n'.join(stuff)))

    # stuff0 = '<{}>{}<{}>'.format(start, matching_lines[0], stop)
    stuff1 = re.sub(r'(c.color\S\S\S\S\S\S)', lambda x: 'c color="{}"'.format(x[0][-6:]), matching_lines[0])
    stuff2 = re.sub(r'(<\d\d:\d\d:\d\d\.\d\d\d>)', lambda x: '<timestamp t="{}"></timestamp>'.format(x[0][1:-1]),
                    stuff1)

    # We need to attach all the CSS tags uptop
    stuff3 = re.sub(r'(c.color\S\S\S\S\S\S)', lambda x: 'c color="{}"'.format(x[0][-6:]),
                    ''.join(re.findall(r'(</?c.*?>)', stuff[0]))) + stuff2

    soup = BeautifulSoup(stuff3, 'lxml')
    children = next(next(soup.children).children)

    words = []
    timesteps = []
    cur_start = start_ts
    conf = 'CCCCCC'
    conf_meanings = {'CCCCCC': 0, 'E5E5E5': 1}

    for child in children.recursiveChildGenerator():
        name = getattr(child, "name", None)
        if name == 'timestamp':
            cur_start = child.attrs['t']
        elif name == 'c':
            if 'color' in child.attrs:
                conf = child.attrs['color']
        elif child.isspace is not None and not child.isspace():
            words.append(child.strip())
            timesteps.append(cur_start)
    timesteps.append(stop_ts)
    buffer = []
    for w_i, word in enumerate(words):
        buffer.append((word, ts_to_sec(timesteps[w_i]), ts_to_sec(timesteps[w_i+1])))
    return buffer


def read_uploaded_vtt(stuff):
    """
    Reads in a user uploaded VTT file
    :param stuff: list of things
    :return:
    """
    start = None
    stop = None
    buffer = []
    everything = []

    def _pop_buffer(start, stop):
        # We have to guess word level alignments, now the buffer looks like
        # ['MALE SPEAKER: And your hand', "shakes from Parkinson's?", '']
        clean_buffer = re.sub(r'<.*?>', '', ' '.join(buffer))

        clean_buffer = [x.strip() for x in clean_buffer.split(' ')]
        clean_buffer = [x for x in clean_buffer if len(x) > 0]

        start_sec = ts_to_sec(start)
        end_sec = ts_to_sec(stop)

        # The timestamps are when the word STARTS which is why I do the +1 thingy'
        # TOTALLY NOT TESTED
        timestamps = np.linspace(start=start_sec, stop=end_sec, num=len(clean_buffer) + 1)
        for b_i, t_s, t_e in zip(clean_buffer, timestamps[:-1], timestamps[1:]):
            everything.append((b_i, t_s, t_e))

    for line in stuff:
        # Sometimes things end with "line:0%" or they have HTML in them
        possible_re_match = re.findall(r'^(.+) --> ([^\s]+)', line)
        if len(possible_re_match) == 1:
            if (start is not None) and (stop is not None):
                _pop_buffer(start, stop)
            # Do it again, on the trimmed line
            possible_re_match = re.findall(r'^(.+) --> (.+)', line[:len("00:00:17.683 --> 00:00:19.285")])
            start, stop = possible_re_match[0]
            buffer = []
        else:
            buffer.append(line)

    if (len(buffer) >= 0) and (start is not None) and (stop is not None):
        _pop_buffer(start, stop)
    return everything

def read_vtt_text(stuff, skip_if_no_timing_info=False):
    """
    Reads in a VTT (as text), split into lines
    :param stuff: LIST of strings
    :param skip_if_no_timing_info: If we can't find any timing information -- skip
    :return: List of tuples
    """

    if skip_if_no_timing_info:
        if '<c>' not in ''.join(stuff):
            return None

    start = None
    stop = None
    buffer = []
    everything = []
    for line in stuff:
        possible_re_match = re.findall(r'^(.+) --> (.+) align:start position:0%', line)
        if len(possible_re_match) == 1:
            if (start is not None) and (stop is not None):
                part = _read_part(buffer, start, stop)
                everything.extend(part)

            start, stop = possible_re_match[0]
            buffer = []
        else:
            buffer.append(line)

    # Add in a missing line
    if len(buffer) > 0:
        try:
            everything.extend(_read_part(buffer, start, stop))
        except (ValueError, KeyError, AttributeError) as e:
            print("Missing line error {}: {}".format(buffer, str(e)), flush=True)
    # The above reads in Google's format. OTHERWISE we need to read a user uploaded format,
    # which is unfortunately differnt
    if (len(everything) == 0) and (len(buffer) > 0) and (buffer[0] == 'WEBVTT'):
        if skip_if_no_timing_info:
            return None
        return read_uploaded_vtt(buffer)
    return everything


def read_vtt(fn):
    """
    Reads in a VTT file and produces a list of tuples, each one containing (word, confidence (0 or 1) and timestamp).
    :param fn: VTT filename
    :return: List of tuples
    """
    with open(fn, 'r') as f:
        stuff = f.read().splitlines()
    return read_vtt_text(stuff)


def channel_to_video_ids(channel_id):
    """
    Converts youtube channel ID to a list of video IDs
    :param channel_id:
    :return:
    """
    ydl_opts = {
        'extract_flat': True,
        'dump_single_json': True,
        'quiet': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            hidden_playlist = ydl.extract_info(f'https://www.youtube.com/channel/{channel_id}/videos/')
            if 'entries' not in hidden_playlist:
                raise DownloadError("entries not in hidden playlist")
            # items = ydl.extract_info(hidden_playlist['url'])
    except DownloadError as e:
        print("Oh no! Got " + str(e))
        return []
    # NOTE: I could even get the titles from here!
    return hidden_playlist['entries']


class PermanentFailure(Exception):
    """A failure no later run can fix: the video is private, removed, or geo-blocked.

    Raised rather than returned so download_all.py can tell it apart from a 429 or a
    transient player error and record the id in its blacklist. Anything unrecognised stays
    transient: wrongly retrying costs one request, wrongly blacklisting loses a video for
    good, so the default has to be retry.
    """
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason
        self.message = message

# Matched against the yt-dlp error text, which quotes YouTube's own playability reason
# verbatim rather than using a fixed set of yt-dlp strings. Specific patterns come first so
# they win the label: a terminated account also says "Video unavailable". Deliberately does
# not list age-gating ("Sign in to confirm your age") or bot checks ("Sign in to confirm
# you're not a bot") -- cookies fix the first and waiting fixes the second.
PERMANENT_FAILURES = (
    ('terminated', 'account associated with this video has been terminated'),
    ('geo_blocked', 'not made this video available in your country'),
    ('geo_blocked', 'who has blocked it in your country'),
    ('members_only', 'join this channel to get access'),
    ('members_only', 'members-only'),
    ('private', 'private video'),
    ('removed', 'has been removed'),
    ('removed', 'this video does not exist'),
    ('unavailable', 'video is no longer available'),
    ('unavailable', 'video unavailable'),
)

def permanent_failure_reason(message):
    """Short reason if this yt-dlp error can never succeed, else None."""
    lowered = message.lower()
    for reason, needle in PERMANENT_FAILURES:
        if needle in lowered:
            return reason
    return None

def ydl_download(id, ydl_opts):
    """
    Downloads from YDL but with error handling and shit
    :param ydl_opts:
    :return: True if success!
    :raises PermanentFailure: the video is gone; retrying it is wasted requests
    """
    # This dosent help
    # user_agents = ['Mozilla/5.0 (X11; Linux i686; rv:82.0) Gecko/20100101 Firefox/82.0',
    #                'Mozilla/5.0 (Linux x86_64; rv:82.0) Gecko/20100101 Firefox/82.0',
    #                 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:82.0) Gecko/20100101 Firefox/82.0',
    #                 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:82.0) Gecko/20100101 Firefox/82.0',
    #                 'Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:82.0) Gecko/20100101 Firefox/82.0',
    #                ]
    # youtube_dl.utils.std_headers['User-Agent'] = user_agents[int(np.random.choice(len(user_agents)))]
    #
    # if os.path.exists('/home/rowan/cookies.txt'):
    #     ydl_opts['cookiefile'] = '/home/rowan/cookies.txt'

    for i in range(2):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(id, download=True, ie_key='Youtube')

                # With ignoreerrors, extract_info returns None instead of raising, and
                # subscripting it was the source of "'NoneType' object is not subscriptable"
                if info is None:
                    return False

                # Download manual subs
                if len(info.get('subtitles') or {}) > 0 and ydl_opts.get('writeautomaticsub'):
                    ydl.params['writesubtitles'] = True
                    # Hack to get manual subs too
                    ps = ydl.process_subtitles(id, normal_subtitles=info['subtitles'], automatic_captions=None)

                    ie = ydl.get_info_extractor(info['extractor_key'])
                    for lang_raw, sub_info in ps.items():
                        sub_lang = f'{lang_raw}-manual'
                        sub_format = sub_info['ext']
                        filename = ydl.prepare_filename(info)
                        sub_filename = subtitles_filename(filename, sub_lang, sub_format, info.get('ext'))
                        try:
                            sub_data = ie._request_webpage(
                                    sub_info['url'], info['id'], note=False).read()
                            with io.open(encodeFilename(sub_filename), 'wb') as subfile:
                                subfile.write(sub_data)
                        except (ExtractorError, IOError, OSError, ValueError) as err:
                            print(str(err), flush=True)
                            continue
            return True
        except DownloadError as e:
            message = str(e)
            permanent = permanent_failure_reason(message)
            if "Too Many Requests" in message:
                # The original slept 24-48 HOURS here. That runs on a worker thread, so a
                # handful of 429s silently parked the whole pool for a day. Fail fast
                # instead: the id lands in the failure list and a later run retries it,
                # since the skip logic is per artifact.
                print(f"Hit a too many requests on {id}, giving up on it for this run", flush=True)
                time.sleep(5)
                return False
            elif permanent:
                raise PermanentFailure(permanent, message)
            else:
                # Retry, which the range(2) loop above was always meant to allow: which
                # player client answers, and whether its JS challenge solves, varies between
                # runs, so the same id can fail once and then succeed. Every branch here used
                # to return False, so the loop could never run twice -- and the branch that
                # printed "RETRYING IN A SEC" was dead anyway, matching 'requested format not
                # available' against a message that reads 'Requested format is not available'.
                print("Oh no! Problem \n\n{}\n".format(str(e)), flush=True)
                time.sleep(5)
        except Exception as e:
            print("Misc exception: {}".format(str(e)))
            return False
    return False


def download_transcript(id, cache_path):
    """
    Given an ID, download and read JUST the transcript + info

    :param id: Video id
    :param cache_path: Where to download the transcript
    :return: Transcript (list of tuples), Info (json)
    """
    info_path = os.path.join(cache_path, f'{id}.v2.info.json')
    if not os.path.exists(info_path):
        # Cookies last, for the same reason as download_video: signed-in cookies make the
        # player answer "The page needs to be reloaded" for most videos. This used to pass
        # them on the only attempt, so every transcript failed whenever --cookies_* was set.
        for use_cookies in (False, True):
            # 'ratelimit' was dropped here: it caps transfer bandwidth, not request rate, so
            # it does nothing about 429s and only slows the (tiny) .vtt transfers.
            ydl_opts = _ydl_opts(
                use_cookies=use_cookies,
                writedescription=False,
                writeinfojson=True,
                write_all_thumbnails=False,
                writeautomaticsub=True,
                writesubtitles=False,
                subtitlesformat='vtt',
                cachedir=cache_path,
                # No 'format' here. This call only fetches subtitles and the info json, but
                # yt-dlp resolves the format selector before skip_download takes effect, so
                # a selector that matches nothing is a hard error. 'best[height=360]' matches
                # nothing now: YouTube serves 360p only as separate video+audio, which is why
                # download_video uses a bv*+ba ladder. It failed on every id.
                outtmpl=os.path.join(cache_path, '%(id)s.v2.%(ext)s'),
                skip_download=True,
                # Regex, not a literal list: YouTube now labels tracks 'en-<trackid>'
                # (e.g. 'en-nP7-2PuUl7o'), which no exact 'en'/'en-US' entry matches.
                subtitleslangs=['en.*'],
                source_address='0.0.0.0',
                no_warnings=True,
            )
            if ydl_download(id, ydl_opts) and os.path.exists(info_path):
                break
            if not _cookies_configured():
                break  # the cookie attempt would be identical to the one just made
        else:
            return [], {}
        # ydl_download can report success without writing the info json
        if not os.path.exists(info_path):
            return [], {}

    transcript = []
    if os.path.exists(os.path.join(cache_path, '{}.v2.en.vtt'.format(id))):
        try:
            transcript = read_vtt(os.path.join(cache_path, '{}.v2.en.vtt'.format(id)))
        except KeyError as e:
            print(f"Oh no error in read_vtt! {id} {e}", flush=True)

    with open(os.path.join(cache_path, f'{id}.v2.info.json'), 'r') as f:
        info = json.load(f)
    return transcript, info


# Pacing shared by download_video and download_transcript, which each build their own
# ydl_opts. YouTube answers unpaced parallel requests with HTTP 429 and then escalates to
# "Sign in to confirm you're not a bot", so requests are spaced and retried with backoff.
# Cookies are set by download_all.py; both None means unauthenticated. COOKIES_FROM_BROWSER
# takes 'browser' or 'browser:profile' -- without a profile, yt-dlp reads the browser's
# default profile, which is the wrong account if the throwaway is a secondary profile.
COOKIES_FROM_BROWSER = None
COOKIES_FILE = None

# Anything smaller than this is a stub or an error page, not a video
MIN_VIDEO_BYTES = 10 * 1024

def _is_playable(path):
    """True if ffprobe finds a video stream and a positive duration.

    A size check alone accepts truncated downloads. siq2 mode caught those only because
    ffmpeg choked while trimming; siq2-full copies the file verbatim, so a corrupt video
    would land in the dataset with nothing downstream to notice.
    """
    try:
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=height:format=duration',
             '-of', 'default=nw=1:nk=1', path],
            capture_output=True, text=True, timeout=120)
        values = [float(v) for v in probe.stdout.split() if v.replace('.', '', 1).isdigit()]
        return probe.returncode == 0 and len(values) >= 2 and min(values) > 0
    except Exception:
        return False

THROTTLE_OPTS = {
    'sleep_interval': 1,
    'max_sleep_interval': 5,
    'extractor_retries': 3,
    'retries': 3,
}

def _cookies_configured():
    return bool(COOKIES_FILE or COOKIES_FROM_BROWSER)

def _ydl_opts(use_cookies=True, **overrides):
    """Merge pacing, cookies, and per-call options into one ydl_opts dict."""
    opts = dict(THROTTLE_OPTS)
    if use_cookies:
        if COOKIES_FILE:
            opts['cookiefile'] = COOKIES_FILE
        elif COOKIES_FROM_BROWSER:
            browser, _, profile = COOKIES_FROM_BROWSER.partition(':')
            opts['cookiesfrombrowser'] = (browser, profile) if profile else (browser,)
    opts.update(overrides)
    return opts

def download_video(id, cache_path, no_download):
    """
    Given an ID, download the video using HQ settings. (might need to turn this down, idk)

    :param id: Video id
    :param cache_path: Where to download the video
    :return: The file that we downloaded things to
    """
    path = os.path.join(cache_path, f'{id}.mp4')

    # Signed-in youtube.com cookies make the player answer "The page needs to be reloaded"
    # for many videos, so try unauthenticated first. Cookies are still the only route to
    # age-gated content, so they are the fallback rather than the default.
    for use_cookies in (False, True):
        ydl_opts = _ydl_opts(
            use_cookies=use_cookies,
            writedescription=False,
            writeinfojson=False,
            skip_download=no_download,
            write_all_thumbnails=False,
            writeautomaticsub=False,
            cachedir=cache_path,
            # YouTube no longer serves a progressive 360p mp4 for many videos, only DASH
            # (separate video+audio). Step the ceiling up rather than jumping straight to
            # best available, which would drop full-length 1080p files onto disk.
            format=('bv*[height<=360]+ba/b[height<=360]/'
                    'bv*[height<=720]+ba/b[height<=720]/bv*+ba/b'),
            merge_output_format='mp4',
            outtmpl=os.path.join(cache_path, '%(id)s.%(ext)s'),
            # ignoreerrors was dropped: it made yt-dlp swallow its own failures, so this
            # returned a path to a file that was missing or truncated. Downstream that
            # showed up as ffmpeg exit 183 (AVERROR_INVALIDDATA) or 254 (ENOENT).
            sub_lang='en',
            source_address='0.0.0.0',
        )
        # Verify something usable landed rather than trusting the return value.
        if ydl_download(id, ydl_opts) and os.path.exists(path) \
                and os.path.getsize(path) > MIN_VIDEO_BYTES and _is_playable(path):
            return path

        # clear partial output so the retry cannot be skipped as already-downloaded
        for stale in glob.glob(os.path.join(cache_path, id + '.mp4*')):
            os.remove(stale)
        if not _cookies_configured():
            break  # the cookie attempt would be identical to the one just made
    return None
