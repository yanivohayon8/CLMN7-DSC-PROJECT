from flask import Flask, escape, request, jsonify,make_response

from src.models.labeling import getLabeledDivision
from src.models.pipeline import pipeline
from src.data.DownloadTranscripts import DownloadTranscript

app = Flask(__name__)

@app.route('/<string:videoURL>', methods=['GET'])
def cut(videoURL):
    # get the parameters from config file
    transcripts, video_id= DownloadTranscript.get_transcript(videoURL)
    slicing_method='sliding_window'
    window_size=40
    step_size_sd=20
    silence_threshold=-30
    slice_length=1000
    step_size_audio=10
    wav_file_path=None
    video_path=None
    wanted_frequency=15
    wanted_similarity_percent = 75
    vector_method='tfidf'
    vectorizing_params=None
    similarity_method='cosine'
    filter_params={'filter_type':None,'mask_shape':None,'sim_thresh':0.4,'is_min_thresh':True}
    clustering_params={'algorithm':'spectral_clustering','n_clusters':13}
    accurrcy_shift=15
    
    time_division, topics = pipeline.run_classification(transcripts,slicing_method,silence_threshold,
                 slice_length,step_size_audio, vector_method,
                 vectorizing_params, similarity_method,filter_params,
                 clustering_params, accurrcy_shift)

    labeldDivision = getLabeledDivision(time_division, topics)
    print(labeldDivision)
    return jsonify(labeldDivision)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)