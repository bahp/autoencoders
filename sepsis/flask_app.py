# Libraries
import os
import sys
import json
import numpy as np
import pandas as pd

# Flask
from flask import Flask
from flask import request
from flask import redirect
from flask import jsonify
from flask import render_template
from flask import send_from_directory

# Others
from sklearn.neighbors import KDTree
from tableone import TableOne
from pathlib import Path

#sys.path.insert(0, os.path.abspath('../..'))

# Create the app.
app = Flask(__name__,
    template_folder=Path('./utils/apps/flask/templates/'),
    static_folder=Path('./objects/'))

# ------------------------------------------------------
# Render Pages
# ------------------------------------------------------
@app.route('/')
def page_index():
    """Displays all workbenches."""
    return render_template('index.html',
        workbenches=sorted([f for f in PATH.iterdir()
            if f.is_dir()]))

@app.route('/workbench/')
def page_workbench():
    """Displays all thumbnails within a workbench."""
    # Get Path
    path = request.args.get('path')
    # Get thumbnails
    thumbnails = sorted(list(Path(path).glob('**/*.jpg')))
    # Return
    return render_template('workbench.html',
        title=path, thumbnails=thumbnails)

@app.route('/embedding/')
def page_embedding():
    """"""
    # Get Path
    path = request.args.get('path')
    data = pd.read_csv(Path(path).parent / 'data.csv')
    return render_template('embedding.html',
        title=path, data=data.to_dict(orient='records'))

def api_data():
    """"""
    pass


if __name__ == "__main__":

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------
    # Specific
    from pathlib import Path

    # Main folder
    PATH = Path('./objects/results/')

    # ---------------------------------------------------
    # Run app
    # ---------------------------------------------------
    # General
    import os

    # .. note: This is necessary when deploying to heroku
    #          because the containers asign a random port
    #          which is passed as environment variable.
    # Get port
    port = int(os.environ.get("PORT", 5555))

    # Run app
    app.run(host='0.0.0.0',
            port=port,
            debug=True,
            use_reloader=False)