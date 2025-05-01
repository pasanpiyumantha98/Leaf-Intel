import React, { useState, useEffect, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios"; // or import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';
import '../App.css';
import { useParams } from 'react-router-dom';
import { Link,useNavigate } from 'react-router-dom';

export default function SimpleImageUploader() {

const Params = useParams();

const [counter, setCounter] = useState(() => {
  const saved = localStorage.getItem("counter");
  return saved ? parseInt(saved, 10) : 1;
});


const navigate = useNavigate();

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [data, setData] = useState(null);
  const [grade, setGrade] = useState("");
  const [points, setPoints] = useState("");
  const [Lot, setLot] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [modalShow, setModalShow] = useState(false);
  const [ready, setReady] = useState("notok");

  // Example endpoint, replace with your API or process.env.REACT_APP_API_URL
  const API_URL = "http://127.0.0.1:8000/predict";

  // Handler for drag-and-drop or file selection
  const onDrop = useCallback((acceptedFiles) => {
    if (!acceptedFiles || acceptedFiles.length === 0) {
      setSelectedFile(null);
      setData(null);
      return;
    }
    // We'll always take the first file if multiple are dropped
    setSelectedFile(acceptedFiles[0]);
    setData(null);
  }, []);

  // Initialize react-dropzone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: "image/*",
    onDrop,
  });

  // Create preview whenever selectedFile changes
  useEffect(() => {
    if (!selectedFile) {
      setPreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    // Clean up the object URL when done
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  // Send file to the server
  const sendFile = async () => {
    if (!selectedFile) return;
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await axios.post(API_URL, formData);
      if (res.status === 200) {
        setData(res.data);
        

      const lotid = parseInt(Params.lotid);
      const response = await axios.get(`/api/lotsfind/${lotid}`);

      //set points
      let point =0;
      
      if(res.data.class==="a")
      {
        point =  response.data.AccQty*25;
      } else if (res.data.class==="b")
      {
        point =  response.data.AccQty*15;
      }
      setPoints(point);

      //check no of samples needed
     
    
      if(response.data.AccQty<700)
      {
        setReady("ok")
        setGrade(res.data.class);
      } else if (counter===2)
      {
        setReady("ok")
        setGrade(res.data.class);
        localStorage.setItem('counter', 1);

      } else
      {
        if(res.data.class==="a" || res.data.class==="b") 
        {
          setReady("reTake")

          const incrementCounter = () => {
            setCounter((prevCounter) => {
              const newCounter = prevCounter + 1;
              localStorage.setItem("counter", newCounter); 
              return newCounter;
            });
          };

          incrementCounter();

        }
      }

      
        
      }

      



    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  // Trigger file upload whenever we have a new preview
  useEffect(() => {
    if (preview) {
      sendFile();
    }
  }, [preview]);

  // Clear all data
  const clearData = () => {
    setSelectedFile(null);
    setPreview(null);
    setData(null);
    setIsLoading(false);
  };

  // Compute confidence in percentage
  let confidence = 0;
  if (data && data.confidence) {
    confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  }


  async function submitGrade(){

    const lotid = parseInt(Params.lotid);
  
    const response = await axios.post(`/samplecheck/update`, {grade:grade,lotid:lotid,points:points});
      
      

    const Lot = await axios.get(`/api/lot/${lotid}`);

    const LotData = Lot.data;

    setLot(LotData);
    


      setModalShow(true);
    } 

   
    

  return (
  



<div className="app-container">
    <Header/>
  <main className="content">

     {/* Bootstrap Modal */}
 {modalShow && (
        <div className="modal fade show" tabIndex="-1" style={{ display: 'block' }} aria-hidden="false">
          <div className="modal-dialog">
            <div className="modal-content">
              <div className="modal-header">
                <h5 className="modal-title"> The lot was successfully accepted!</h5>
                <button
                  type="button"
                  className="btn-close"
                  data-bs-dismiss="modal"
                  aria-label="Close"
                  onClick={() => {
                    setModalShow(false);
                    navigate('/accepting'); 
                  }}
                ></button>
              </div>
              <div className="modal-body">
                <p>Lot ID: {Lot.LotId}</p>
                <p>Grade: {Lot.Grade}</p>
                <p>Quantity: {Lot.AccQty}</p>
                <p>Supplier: {Lot.SuppId}</p>

               
              </div>
              <div className="modal-footer">
              <button
                  type="button"
                  className="btn btn-secondary"
                  data-bs-dismiss="modal"
                  onClick={() => {
                    setModalShow(false);
                    navigate('/accepting'); 
                  }}
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        </div>
      )}


    <h1 class="pagetitle">Sample Check</h1>


<div class="row">

  <div class="col-sm-2">

 

  </div>

  <div class="col-sm-8">
      
  <div style={{ maxWidth: 600, margin: "20px auto", textAlign: "center" }}>
      

      {/* Drop Zone or "Select File" area */}
      {!preview && (
        <div
          {...getRootProps()}
          style={{
            border: "2px dashed #ccc",
            borderRadius: 8,
            padding: 40,
            cursor: "pointer",
            marginBottom: 20,
          }}
        >
          <input {...getInputProps()} />
          {isDragActive ? (
            <p>Drop your image here ...</p>
          ) : (
            <p>Drag 'n' drop an image, or click to select</p>
          )}
        </div>
      )}

      {/* Preview of the image, if present */}
      {preview && (
        <div style={{ marginBottom: 20 }}>
          <img
            src={preview}
            alt="preview"
            style={{
              maxWidth: "250px",
              height: "300px",
              display: "block",
              margin: "0 auto",
            }}
          />
        </div>
      )}

      {/* Loading indicator */}
      {isLoading && <p>Processing...</p>}

      {/* Display classification results if available */}
      {data && (
        <div style={{ marginTop: 20 }}>
          <p>
            <strong>Grade: </strong>
            {data.class}
          </p>
          
          <p>
            <strong>Confidence: </strong>
            {confidence}%
          </p>
          <p>
            <strong>Points: </strong>
            {points}
          </p>
          <p>
            
            {data.class === "a" ? "This sample is good for production" : ""}
            {data.class === "b" ? "This sample is okay for production" : ""}
            {data.class === "c" ? "This sample is not suitable for Production. " : ""}
            {ready==="reTake" ? "\t Since the lot weight is higher, perform the quality check 1 more time." : ""}
			{data.class === "uk" ? "TRY AGAIN" : ""}
          </p>
        </div>
      )}

{   
      
      (preview && (grade === "a" || grade === "b") && ready === "ok") && (
        <button class="btn btn-success" onClick={submitGrade} style={{ marginTop: 20 }}>
          Accept
        </button>
      )}
       &emsp; 
      {/* Clear button appears only when there's data or a preview */}
      {(preview || data) && (
        <button class="btn btn-danger" onClick={clearData} style={{ marginTop: 20 }}>
          Re Take
        </button>
      )}

      <br></br><br></br>
    </div>

  </div>

  <div class="col-sm-2">

   
  




   </div>
  </div>
  
  






  </main>
</div>

  );
}
