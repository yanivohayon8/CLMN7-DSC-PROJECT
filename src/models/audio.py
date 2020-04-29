#!/usr/bin/env python
# coding: utf-8

# In[15]:


from pydub import AudioSegment
from pydub.silence import split_on_silence

def getSubjSwitchPoints(wavFilePath):
    switchPoints = []
    audio_segment = AudioSegment.from_wav(wavFilePath)
    
    


# In[78]:


from __future__ import unicode_literals

get_ipython().system('pip install --upgrade youtube-dl')
get_ipython().system('pip install --upgrade pygame')
get_ipython().system('pip install --upgrade pydub')
get_ipython().system('pip install --upgrade ffprobe')
get_ipython().system('pip install --upgrade ffmpeg')


import youtube_dl

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=MkiUBJcgdUY'])


# In[80]:


# Import the AudioSegment class for processing audio and the 
# split_on_silence function for separating out silent chunks.
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
   ''' Normalize given audio chunk '''
   change_in_dBFS = target_dBFS - aChunk.dBFS
   return aChunk.apply_gain(change_in_dBFS)

# Load your audio.
audio = AudioSegment.from_wav("Mod-01 Lec-01 Foundation of Scientific Computing-01-MkiUBJcgdUY.wav")
print(audio)


# In[81]:


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


# In[82]:


audio_len = len(audio) # result is in ms

SLICE_LEN = 1000 # ms   --- slice the audio to secondes

# find silence and add start and end indicies to the to_cut list
silence_starts = []

last_slice_start_point = audio_len - SLICE_LEN
slice_start_points = range(0, last_slice_start_point + 1, 1)
print(slice_start_points)

silence_thresh = -30 # silence threshold

# convert silence threshold to a float value (so we can compare it to rms)
silence_thresh = db_to_float(silence_thresh) * audio.max_possible_amplitude
print(silence_thresh)

for i in slice_start_points:
    audio_slice = audio[i:i + SLICE_LEN]
    #print(audio_slice.rms)
    if audio_slice.rms <= silence_thresh:
        silence_starts.append(i)


print(len(silence_starts))


# In[69]:


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

print(silent_ranges)

silence_ranges_len = list(map(lambda silence_range : silence_range[1]-silence_range[0] ,silent_ranges))

print(silence_ranges_len)

avg_silence_time = sum(list(silence_ranges_len))/len(list(silence_ranges_len))
print(avg_silence_time)

subject_switch_silent_ranges=[]

for idx,silence_range in enumerate(silence_ranges_len):
   if silence_range > avg_silence_time:
       subject_switch_silent_ranges.append(silent_ranges[idx])

       
print(subject_switch_silent_ranges)


# In[83]:


from pydub import AudioSegment
from pydub.silence import split_on_silence

def getSubjSwitchPoints(wavFilePath , silence_threshold):
    switchPoints = []
    audio = AudioSegment.from_wav(wavFilePath)
    audio_len = len(audio) # result is in ms

    SLICE_LEN = 1000 # ms   --- slice the audio to secondes

    # find silence and add start and end indicies to the to_cut list
    silence_starts = []

    last_slice_start_point = audio_len - SLICE_LEN
    slice_start_points = range(0, last_slice_start_point + 1, 1)

    silence_thresh = silence_threshold # silence threshold

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

        
    return subject_switch_silent_ranges
    


# In[72]:


#test
from __future__ import unicode_literals

get_ipython().system('pip install --upgrade youtube-dl')
get_ipython().system('pip install --upgrade pygame')
get_ipython().system('pip install --upgrade pydub')
get_ipython().system('pip install --upgrade ffprobe')
get_ipython().system('pip install --upgrade ffmpeg')


import youtube_dl

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=Akwm2UZJ34o'])
    


# In[86]:


silent_ranges = getSubjSwitchPoints("Mod-01 Lec-01 Foundation of Scientific Computing-01-MkiUBJcgdUY.wav")



print(silence_ranges_len)


# In[90]:


silence_ranges_len= list(map(lambda silence_range : silence_range[1]-silence_range[0] ,silent_ranges))
print(silent_ranges)
print(silence_ranges_len)


# In[93]:


import matplotlib.pyplot as plt

n, bins, patches = plt.hist(x=silence_ranges_len, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('ms')
plt.ylabel('quantity')
plt.title('My Very Own Histogram')
plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


# In[ ]:




