from flask import Flask, render_template, Response, jsonify
from main import my_function

my_function()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

# def gen(camera):
#     while True:
#         global df1
#         frame, df1 = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response((my_function()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# if __name__ == '__main__':
#     app.debug = True
#     app.run()


if __name__ == '__main__':
    app.run(debug=True, port=8000)
