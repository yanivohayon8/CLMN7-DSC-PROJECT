from flask import Flask, escape, request, jsonify,make_response
from src.models.pipeline import pipeline
from src.data.DownloadTranscripts import DownloadTranscript
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/<string:videoURL>', methods=['GET'])
def cut(videoURL):
    # get the parameters from config file
    transcripts= DownloadTranscript.get_transcript(videoURL)
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
    clustering_params={'algorithm':'spectral_clustering','n_clusters':4}
    accurrcy_shift=15
    # get the transcript
    
    divition = pipeline.run_classification(transcripts,slicing_method=slicing_method,silence_threshold=silence_threshold,
                 slice_length=slice_length,step_size_audio=step_size_audio, vector_method=vector_method,
                 vectorizing_params=vectorizing_params, similarity_method=similarity_method,filter_params=filter_params,
                 clustering_params=clustering_params, accurrcy_shift=accurrcy_shift)
    print (divition)
    response = jsonify(divition);
    response.headers["Content-type"] = "application/json"
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers["Access-Control-Expose-Headers"] = 'Access-Control-Allow-Origin'
    response.headers["Access-Control-Allow-Headers"] = "Origin, X-Requested-With, Content-Type, Accept"
    response.headers["Access-Control-Allow-Methods"] = "GET"
    return response

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)