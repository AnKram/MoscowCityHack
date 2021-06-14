from flask import Flask, request, render_template, jsonify, send_from_directory
#from hack_back import getPredict
from functions import get_data

#from flask_json import FlaskJSON, JsonError, json_response, as_json
#from flask_cors import CORS, cross_origin

app = Flask(__name__, 
            template_folder='htmls', 
            static_url_path='', 
            static_folder='static')

#cors = CORS(app)
#FlaskJSON(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['JSON_AS_ASCII'] = False


@app.route('/parse', methods=['POST'])
#@cross_origin()
def get_text():
    data = request.json
    result = get_data(data['cv'])
    #a = {f'!!!{data}!!!': 'sdfdsf'}
    return jsonify(result)


@app.route('/test')
def render_test_page():
    return render_template('test2.html')


@app.route('/view')
def render_view():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4355)
