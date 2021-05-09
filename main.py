from flask import Flask, render_template, request
import secrets
import os
import colourmodel
import base64
import urllib.request

__DIR__ = os.path.dirname(__file__)

app = Flask(__name__)


# Create directories
if not os.path.exists(os.path.join(__DIR__, "static/img/uploads")):
    os.makedirs(os.path.join(__DIR__, "static/img/uploads"))
if not os.path.exists(os.path.join(__DIR__, "static/img/converted")):
    os.makedirs(os.path.join(__DIR__, "static/img/converted"))


@app.route('/')
def index():
    return render_template('index.html')


app.config["IMAGE_UPLOADS"] = os.path.join(__DIR__, "static/img/uploads")
app.config["IMAGE_FILETYPES"] = ["PNG", "JPG", "JPEG"]


def imageExtensionCheck(filename):
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1]

    if extension.upper() in app.config["IMAGE_FILETYPES"]:
        return extension

    else:
        return False


@app.route('/upload-image', methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":

        if request.form['request'] == "upload" or request.form['request'] == "url":

            if request.form['request'] == "upload":
                if 'file' not in request.files:
                    error = "No image has been selected."
                    return {"error": error}

                image = request.files["file"]

                if image.filename == "":
                    error = "Image must have a filename."
                    return {"error": error}

                if not imageExtensionCheck(image.filename):
                    error = "Image must be of the filetype: .jpg, .jpeg or .png"
                    return {"error": error}

                extension = imageExtensionCheck(image.filename)

                filename = secrets.token_hex(nbytes=16) + "." + extension
                print("Generated filename : " + filename)

                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
                print("Save complete")
            else :
                try:
                    image = urllib.request.urlopen(request.form['url'])
                except:
                    error = "That image doesn't exist or cannot be read"
                    return {"error": error}

                if not imageExtensionCheck(request.form['url']):
                    error = "Image must be of the filetype: .jpg, .jpeg or .png"
                    return {"error": error}

                extension = imageExtensionCheck(request.form['url'])

                filename = secrets.token_hex(nbytes=16) + "." + extension
                print("Generated filename : " + filename)

                output = open(os.path.join(app.config["IMAGE_UPLOADS"], filename), "wb")
                output.write(image.read())
                output.close()

                print("Save complete")

            converted_path = colourmodel.conversion(filename)
            converted_image = open(os.path.join(__DIR__, converted_path), 'rb')

            after_string = base64.encodebytes(converted_image.read()).decode('utf-8')

            print("Conversion complete")

            return {"response": True, "after": after_string, "url": converted_path, "filename": filename}

        else:
            error = "Invalid upload type."
            return {"error": error}

    else:

        error = "Must be a POST request"
        return {"error": error}


if __name__ == "__main__":
    app.run(debug=True)
