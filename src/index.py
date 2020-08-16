from flask import Flask, escape, request, jsonify,make_response

from src.models.labeling import getLabeledDivision
from src.models.pipeline import pipeline
from src.data.DownloadTranscripts import DownloadTranscript
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/<string:videoURL>', methods=['GET'])
def cut(videoURL):
    # get the parameters from config file
    transcripts, video_id= DownloadTranscript.get_transcript(videoURL)
    slicing_method='sliding_window'
    window_size=120
    step_size_sd=15
    vector_method='tfidf'
    vectorizing_params=None
    similarity_method='cosine'
    filter_params={'filter_type':"median",'mask_shape':(2,2),'sim_thresh':0.6,'is_min_thresh':True}
    clustering_params={'algorithm':'spectral_clustering','n_clusters':30}
    accurrcy_shift=15
    
    time_division, topics = pipeline.run_classification(transcripts,slicing_method,window_size=window_size,
                 step_size_sd=step_size_sd, vector_method=vector_method,
                 vectorizing_params=vectorizing_params, similarity_method=similarity_method,filter_params=filter_params,
                 clustering_params=clustering_params, accurrcy_shift=accurrcy_shift)

    labeldDivision = getLabeledDivision(time_division, topics)
    print(labeldDivision)
    response = jsonify(labeldDivision);
    response.headers["Content-type"] = "application/json"
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers["Access-Control-Expose-Headers"] = 'Access-Control-Allow-Origin'
    response.headers["Access-Control-Allow-Headers"] = "Origin, X-Requested-With, Content-Type, Accept"
    response.headers["Access-Control-Allow-Methods"] = "GET"
    return response

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)
