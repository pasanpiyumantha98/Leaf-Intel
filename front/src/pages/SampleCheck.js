import React, { useState, useRef, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';
import '../App.css';
import { useParams, useNavigate } from 'react-router-dom';

export default function WebcamSampleCheck() {
  const { lotid } = useParams();
  const navigate = useNavigate();
  const webcamRef = useRef(null);

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [data, setData] = useState(null);
  const [grade, setGrade] = useState("");
  const [points, setPoints] = useState(0);
  const [Lot, setLot] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [modalShow, setModalShow] = useState(false);
  const [ready, setReady] = useState("notok");
  const [phone, setPhone] =useState('');

  const [counter, setCounter] = useState(() => {
    const saved = localStorage.getItem("counter");
    return saved ? parseInt(saved, 10) : 1;
  });

// Retriving manager phone number to inform updates
  useEffect(()=>{

    const loadData = async()=>{

      const response2 = await axios.get(`/api/manager/phone`);
      
      const pn = response2.data.Phone;

      setPhone(pn)

    };

    loadData();

  },[]);

  const API_URL = "http://127.0.0.1:8000/predict";

  const capture = useCallback(() => {
    const screenshot = webcamRef.current.getScreenshot();
    setImage(screenshot);
    setData(null);
  }, [webcamRef]);

  useEffect(() => {
    if (image) {
      const objectUrl = URL.createObjectURL(dataURItoBlob(image));
      setPreview(objectUrl);
      sendFile();
      return () => URL.revokeObjectURL(objectUrl);
    }
  }, [image]);

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

  const sendFile = async () => {
    if (!image) return;
    setIsLoading(true);

    try {
      const blob = dataURItoBlob(image);
      const formData = new FormData();
      formData.append("file", blob, "webcam_capture.jpg");

      const res = await axios.post(API_URL, formData);
      if (res.status === 200) {
        setData(res.data);
        const response = await axios.get(`/api/lotsfind/${lotid}`);
        let point = 0;
        if (res.data.class === "a") point = response.data.AccQty * 25;
        else if (res.data.class === "b") point = response.data.AccQty * 15;
        setPoints(point);

        if (res.data.class === "a" || res.data.class === "uk"){

          async function handleSend() {
               
            let to=phone
            let body =`The lot ${lotid} has failed the sample check!`
            
            try {
              const res = await axios.post('/whatsapp-send/go',
                { to, body }
              );
              
            } catch (err) {
              console.error(err);
              
            }
          }
    
          await handleSend()

        } 

        if (response.data.AccQty < 700 || counter === 2) {
          setReady("ok");
          setGrade(res.data.class);
          if (counter === 2) localStorage.setItem('counter', 1);
        } else {
          if (res.data.class === "a" || res.data.class === "b") {
            setReady("reTake");
            const newCounter = counter + 1;
            setCounter(newCounter);
            localStorage.setItem("counter", newCounter);
          }
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const submitGrade = async () => {
    await axios.post(`/samplecheck/update`, { grade, lotid: parseInt(lotid), points });
    const LotData = await axios.get(`/api/lot/${lotid}`);
    setLot(LotData.data);
    setModalShow(true);
  };

  const clearData = () => {
    setImage(null);
    setPreview(null);
    setData(null);
    setIsLoading(false);
  };

  let confidence = 0;
  if (data && data.confidence) confidence = (parseFloat(data.confidence) * 100).toFixed(2);

  return (
    <div className="app-container">
      <Header />
      <main className="content">
        {modalShow && (
          <div className="modal fade show" tabIndex="-1" style={{ display: 'block' }} aria-hidden="false">
            <div className="modal-dialog">
              <div className="modal-content">
                <div className="modal-header">
                  <h5 className="modal-title">The lot was successfully accepted!</h5>
                  <button type="button" className="btn-close" onClick={() => { setModalShow(false); navigate('/accepting'); }}></button>
                </div>
                <div className="modal-body">
                  <p>Lot ID: {Lot.LotId}</p>
                  <p>Grade: {Lot.Grade.toUpperCase()}</p>
                  <p>Quantity: {Lot.AccQty}</p>
                  <p>Supplier: {Lot.SuppId}</p>
                </div>
                <div className="modal-footer">
                  <button type="button" className="btn btn-secondary" onClick={() => { setModalShow(false); navigate('/accepting'); }}>Done</button>
                </div>
              </div>
            </div>
          </div>
        )}

        <h1 className="pagetitle">Sample Check</h1>

        <div className="row">
          <div className="col-sm-2"></div>
          <div className="col-sm-8" style={{ textAlign: 'center' }}>
          <div class="card">
         <div class="card-body">   
            
            {!preview && (
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                style={{ width: 640, height: 480, marginBottom: 20 }}
                videoConstraints={{ width: 640, height: 480, deviceId:"Wed Camera" }}
              />
            )}

            {preview && (
              <div style={{ marginBottom: 20 }}>
                <center><img src={preview} alt="Captured" style={{ maxWidth: "250px", height: "300px", display: "block"}} /></center>
              </div>
            )}

            {!preview ? (
             <center> <button className="btn btn-success" onClick={capture}>Check</button></center>
            ) : (
              <>
                {isLoading && <p>Processing...</p>}

                {data && (
                  <div style={{ marginTop: 20 }}>
                    <p><strong>Grade:</strong> {data.class.toUpperCase()}</p>
                    <p><strong>Confidence:</strong> {confidence}%</p>
                    <p><strong>Points:</strong> {points}</p>
                    <p>
                      {data.class === "a" && "This sample is good for production"}
                      {data.class === "b" && "This sample is okay for production"}
                      {data.class === "c" && "This sample is not suitable for Production."}
                      {ready === "reTake" && " Since the lot weight is higher, perform the quality check 1 more time."}
                      {data.class === "uk" && "TRY AGAIN"}
                    </p>
                  </div>
                )}

                {(grade === "a" || grade === "b") && ready === "ok" && (
                  <button className="btn btn-success" onClick={submitGrade} style={{ marginTop: 20 }}>Accept</button>
                )} &emsp;

                <button className="btn btn-danger" onClick={clearData} style={{ marginTop: 20 }}>Re Take</button>
              </>
            )}
          </div>
          </div>
          </div>
          <div className="col-sm-2"></div>
        </div>
      </main>
    </div>
  );
}
