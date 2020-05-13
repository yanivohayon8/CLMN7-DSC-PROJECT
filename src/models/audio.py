from __future__ import unicode_literals
from pydub import AudioSegment
import youtube_dl
import urllib.parse as urlparse


def db_to_float(db, using_amplitude=True):
    """
    Converts the input db to a float, which represents the equivalent
    ratio in power.
    """
    db = float(db)
    if using_amplitude:
        return 10 ** (db / 20)
    else:  # using power
        return 10 ** (db / 10)

# returns the silent ranges who's length is longer than the average 
# silent range in the audio file.
def getSubjSilentRanges(wav_file_path, silence_threshold, slice_length, step_size):
    SLICE_LEN = slice_length
    STEP_SIZE = step_size
    silence_thresh = silence_threshold
        
    switchPoints = []
    audio = AudioSegment.from_wav(wav_file_path)
    if audio != None:
        audio_len = len(audio) # result is in ms

		# find silence and add start and end indicies to the to_cut list
        silence_starts = []

        last_slice_start_point = audio_len - SLICE_LEN
        slice_start_points = range(0, last_slice_start_point + 1,STEP_SIZE)

		# convert silence threshold to a float value (so we can compare it to rms)
        silence_thresh = db_to_float(silence_thresh) * audio.max_possible_amplitude

        for i in slice_start_points:
            audio_slice = audio[i:i + SLICE_LEN]
			#print(audio_slice.rms)
            if audio_slice.rms <= silence_thresh:	
                silence_starts.append(i)
		
		# combine the silence we detected into ranges (start ms - end ms)
        silent_ranges = []

        prev_i = silence_starts.pop(0)
        current_range_start = prev_i

        for silence_start_i in silence_starts:
            continuous = (silence_start_i == prev_i + 1)

			# sometimes two small blips are enough for one particular slice to be
			# non-silent, despite the silence all running together. Just combine
			# the two overlapping silent ranges.
            silence_has_gap = silence_start_i > (prev_i + SLICE_LEN)

            if not continuous and silence_has_gap:	
                silent_ranges.append([current_range_start,prev_i + SLICE_LEN])
                current_range_start = silence_start_i
            prev_i = silence_start_i

        silent_ranges.append([current_range_start,prev_i + SLICE_LEN])

        silence_ranges_len = list(map(lambda silence_range : silence_range[1]-silence_range[0] ,silent_ranges))

        avg_silence_time = sum(list(silence_ranges_len))/len(list(silence_ranges_len))

        subject_switch_silent_ranges=[]

        for idx,silence_range in enumerate(silence_ranges_len):
            if silence_range > avg_silence_time:
                subject_switch_silent_ranges.append(silent_ranges[idx])

        return list(map(lambda silence_range : int(((silence_range[1]+silence_range[0]) / 2 ) / 1000),subject_switch_silent_ranges))
    else:
        print("video wasn't found")
        return None
		
		
def downloadAudioFromYoutube(youtube_path):
    url_data = urlparse.urlparse(youtube_path)
    #query = urlparse.parse_qs(url_data.query)
    #video_id = query["v"][0]
    ydl_opts = {
		'format': 'bestaudio/best',
		'postprocessors': [{
			'key': 'FFmpegExtractAudio',
			'preferredcodec': 'wav',
			'preferredquality': '192',
			#'outtmpl': './../../data/raw/audio/%(video_id)s.%(ext)s',
            
		}],
        'outtmpl': './../data/raw/audio/%(id)s.%(ext)s' 
        #'outtmpl': './../../data/raw/audio/%(id)s.%(ext)s' 
	}

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_path])