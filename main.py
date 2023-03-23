from flask import Flask, render_template, request
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# generate sample data
X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    # get parameters from request
    k = int(request.form['k'])

    # run KMeans clustering algorithm
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)

     # plot clusters and save to base64-encoded string
    plt.scatter(X[:, 0], X[:, 1], c=clusters)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode()

    # render template with plot
    return render_template('result.html', img_data=img_data)

if __name__ == '__main__':
    app.run(debug=True)