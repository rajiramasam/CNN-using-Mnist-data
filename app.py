from flask import Flask,request,render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image,ImageOps

app=Flask(__name__)

model=load_model('file.hdf5')

def prepare_image(image):
    img=ImageOps.grayscale(image)
    img=ImageOps.invert(img)
    img=img.resize((28,28))
    img=np.array(img)/255.0
    # img = img.astype('float32') / 255.0
    img=img.reshape(1,28,28,1)
    return img

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        file=request.files['file']
        if file:
            image=Image.open(file.stream)
            image=prepare_image(image)
            pred=model.predict(image)
            predicted_cls=np.argmax(pred)
            return render_template('result.html',predicted=predicted_cls)
    return render_template('index.html')
if __name__=='__main__':
    app.run(debug=True,use_reloader=False)