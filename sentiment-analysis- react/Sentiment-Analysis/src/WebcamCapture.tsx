import React, { useRef, useCallback, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { IoIosSend } from "react-icons/io";

const WebcamCapture = ({ onCapture,setIsImageSet }) => {
  const [capturedImage, setCapturedImage] = useState(null);
  const [emotion,setEmotion]=useState(null)

  const webcamRef = useRef(null);

  const capture = useCallback(async () => {
    
    const imageSrc = webcamRef.current.getScreenshot();
    setCapturedImage(imageSrc);
    setIsImageSet(true)

    // Save the captured image locally (you can customize the file name)
    const fileName = 'captured_image.jpg';
    const formData = new FormData();
    formData.append('image', dataURItoBlob(imageSrc), fileName);

    // Send the image as form data to the Flask API
    try {
      const response = await axios.post('http://127.0.0.1:5000/upload_image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('API Response:', response.data);
      onCapture(response.data.emotion)
      setEmotion(response.data.emotion)

    } catch (error) {
      console.error('Error sending image to Flask API:', error);
    }
  }, [webcamRef, onCapture]);

  // Convert data URI to Blob
  const dataURItoBlob = (dataURI) => {
    const byteString = atob(dataURI.split(',')[1]);
    const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    return new Blob([ab], { type: mimeString });
  };

  return (
    <>
      <div className="webcam-img-container">
        <div className="webcam-container">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            style={{ width: '500px', marginBottom:'20px' }}
          />
          <button onClick={capture} >Get Songs Based On Expression <IoIosSend /></button>
        </div>
        <div className="capture-img-container">
        {capturedImage ? (
          <>
            <img src={capturedImage} alt="Captured" className="captured-image" />
            {emotion &&<h2>You are currently {emotion}</h2>}
          </>
        ):
        <h1 style={{textAlign:'center', marginTop:"35%"}}>
          Grab a photo or Type in what you feel like to get playlists recommended based on your mood!
        </h1>

        }
        </div>
      </div>
    </>
  );
};

export default WebcamCapture;
