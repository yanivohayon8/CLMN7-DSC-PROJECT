from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse,parse_qs
from os import path
from pathlib import Path
import json
import ast

class DownloadTranscript():
    @staticmethod
    def get_transcript(id):
        curr_path = Path(__file__)
        transcripts_path = path.join(curr_path.parent.parent.parent, 'data\\raw\\transcripts')

        video_id = id
        #video_id = "h9wxtqoa1jY"
        print ("The video ID is %s" % (video_id))
        t_path = path.join(transcripts_path,video_id + '.json')
        if(path.exists(t_path)):
            with open(t_path,encoding="utf8") as f:
                #transcript =ast.literal_eval(f.read()) 
                transcript =json.load(f) 
        else:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            with open(t_path,'w+') as f:
                json.dump(transcript,f)
        return transcript, video_id