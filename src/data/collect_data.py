# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:56:01 2020

@author: yaniv
"""

# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


# my Imports 
from youtube_transcript_api import YouTubeTranscriptApi
import urllib.parse as urlparse
import json
import traceback
import os
import re

from pyyoutube import Api
from youtube_search import YoutubeSearch




'''@click.command()
#@click.option('--channel-name')
#@click.option('--load_from_file',type=bool)
@click.argument('channel_name')
#def main(channel_name,load_from_file=False):'''
@click.command()
@click.option('-c','--channel-name')
def main(channel_name="YaleCourses",load_from_file=False):
    # find all the videos 
    
    api = Api(api_key="AIzaSyCw0j0aCe0y_T42q3RLBoVtGXlTOMGGaSM")
    # AIzaSyCw0j0aCe0y_T42q3RLBoVtGXlTOMGGaSM
    
    print("Setup dir to save the transcripts of %s channel" %(channel_name))
    channel_dir = os.path.join(raw_dir,"transcripts",channel_name)
    channel_id_file = os.path.join(raw_dir,"video_ids",channel_name + ".txt")
    
    if not os.path.exists(channel_dir):
        os.mkdir(channel_dir)
    else:
        print("\tThe folder of the channel %s is already exist\n"
                "\tdelete it before executing this script -" 
                "we don't want to override your data" %(channel_name))
        return
    '''
        Since google is blocking after a while the retrival of the IDs,
        We will write the IDs to a file as a buffer for safety.
    '''    
    if load_from_file is False:
        print("Retriving %s channel information" % (channel_name))
        channel_by_name=api.get_channel_info(channel_name=channel_name)
        print ("\tFetch all the playlists")
        playlists_by_channel = api.get_playlists(channel_id=channel_by_name.items[0].id,count=None)
        print("\tFetch all the videos of the playlist")
        playlists_videos = []
        for playlist in playlists_by_channel.items:
            print("\t\tFetching videos IDs of playlist %s" %(playlist.id))
            playlists_videos.append(api.get_playlist_items(playlist_id=playlist.id,count=None))
    
        videos_ids = []
        for playlist in playlists_videos:
            for video in playlist.items:
               videos_ids.append(video.snippet.resourceId.videoId) 
        print("We gathered now %s videos, saving save to file" %(len(videos_ids)))
        with open(channel_id_file,'w') as f:
            json.dump(videos_ids,f)
    else:
        with open(channel_id_file,'r') as f:
            videos_ids = json.load(f)
        
    print("Save %s channel videos transcripts" % (channel_name) )
    #map(save_transcript,videos_ids)
    #[save_transcript(vd) for vd in videos_ids]
    
    for video_id in videos_ids:
        print ("The video ID is %s" % (video_id))
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)#,languages=['en']
            #transcript_list = [transcript for transcript in transcript_list\
            #                   if bool(re.match(transcript.language,"[en]*"))]
            video_transcripts = None
            for transcript in transcript_list:
                # the Transcript object provides metadata properties
                print("Video id : ", transcript.video_id)
                print("\tlanguage : %s , language code : %s" %(transcript.language,transcript.language_code))
                print("\tis_generated: %s, is_translatable: %s" %(transcript.is_generated,transcript.is_translatable))
                if transcript.language_code == 'en' and transcript.is_generated is False:
                    actual_transcript = transcript.fetch()
                    video_transcripts = actual_transcript
            
            if video_transcripts is not None:
                #print( "Current length json of trancsript is " ,len(transcript))
                video_path = os.path.join(raw_dir,"transcripts",channel_name,video_id + ".json")
                with open(video_path,'w') as outfile:
                    json.dump(video_transcripts,outfile)    
        except Exception as e:
            print(e)

    print("Finish main")
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    raw_dir = os.path.join(project_dir,"data","raw")
    
    
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
    
    
    
    
    
    
    