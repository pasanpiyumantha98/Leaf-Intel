import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState,useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faInfoCircle, faEnvelope, faUser, faCog, faSuperscript, faPeopleGroup, faClover, faHandFist, faHandHolding, faAd, faAdd, faGlasses, faDeleteLeft } from '@fortawesome/free-solid-svg-icons';
import LotList from '../components/Lot-List';
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";



function Accepting(){


    const [SuppId, setSuppID] = useState("");
    const [Qty, setQty] =useState("");
    const [Range, setRange] =useState("3");
    const [phone, setPhone] =useState('');
    const [Pending, setPending] =useState("5000.09");
    
    const navigate = useNavigate();


    async function submitAccForm()
    {

        const response = await axios.post(`/accepting/first`,{SuppID:SuppId,Qty:Qty,Range:Range})

        const stat = response.data.message;

        const lotid =response.data.lotid;

        if(stat==="OK")
        {
          navigate(`/accepting/samplecheck/v2/${lotid}`); 
        } else if(stat==="SuppNotFound"){
          toast.error("Supplier ID incorrect!");
        } else {
          toast.error("Something went wrong!");
        }
        

    }



    useEffect(()=>{

      const loadData = async()=>{
  
 
        const response = await axios.get(`/api/lots/pending`);
  
        const pending = response.data;
    
        setPending(pending);


        const response2 = await axios.get(`/api/manager/phone`);
        
        const pn = response2.data.Phone;

        setPhone(pn)
  
        
  
  
      };
  
      loadData();
  
  
    },[]);

    async function transfer()
    {
      

        const response = await axios.get(`/api/transfer`)

        const stat = response.data;

        if(stat==="ok")
        {      
          async function handleSend() {
               
            let to=phone
            let body =`The pending lots of ${Pending} Kg has been accepted to production!`
            
            try {
              const res = await axios.post('/whatsapp-send/go',
                { to, body }
              );
              
            } catch (err) {
              console.error(err);
              
            }
          }
    
          await handleSend()
          window.location.reload(); 
        }
        

    }

  

    return(
        <>
  <ToastContainer />
  <div className="app-container">
    <Header/>
  <main className="content">


    <h1 class="pagetitle">Accepting</h1>


<div class="row">

  <div class="col-sm-3">

 

  </div>

  <div class="col-sm-2">

  

  </div>

  <div class="col-sm-2">

   <div class="card">
    <div class="card-body custom-card-body1">
    <a  data-bs-toggle="modal" data-bs-target="#search"> <center> <FontAwesomeIcon icon={faAdd} className="icon" /> </center>
    <h5 class="card-title"><center>New Lot</center></h5> </a>
  

   
    <div className="modal fade" id="search">
  <div className="modal-dialog">
    <div className="modal-content">
      
      <div className="modal-header">
        <h4 className="modal-title">Lot Details</h4>
        <button
          type="button"
          className="btn-close"
          data-bs-dismiss="modal"
        ></button>
      </div>

      <div className="modal-body">
        <label>Supplier ID &ensp; &ensp;</label>
        <input
          className="inputfsmall"
          type="text"
          placeholder="Supplier ID"
          name="ID"
          value={SuppId}
          onChange={(q) => setSuppID(q.target.value)}
          required
        />

        <br />

        <label>Quantity &ensp;&ensp;&ensp; &ensp;</label>
        <input
          className="inputfsmall"
          type="text"
          placeholder="Quantity in KG"
          name="Qty"
          value={Qty}
          onChange={(q) => setQty(q.target.value)}
          required
        />

        <br />

        <label htmlFor="waterLevel">Water Level</label>
<div style={{ display: 'flex', alignItems: 'center' }}>
  <span style={{ marginRight: '8px' }}>Low</span>
  <input
    type="range"
    className="form-range"
    id="waterLevel"
    name="range"
    min="0"
    max="5"
    value={Range}
    onChange={(e) => setRange(e.target.value)}
  />
  <span style={{ marginLeft: '8px' }}>High</span>
</div>


        <br /> <br />

        <div style={{ textAlign: 'center' }}>
          {/* Replaced <a> with <button> to close modal + navigate */}
          <button
            type="button"
            className="btn btn-success success padl"
            data-bs-dismiss="modal"            // closes the modal
            onClick={submitAccForm}            // triggers your submit logic + navigation
          >
            Proceed
          </button>
        </div>
      </div>
    </div>
  </div>
</div>




   </div>
  </div>


  
  </div>

  <div class="col-sm-2">
  <div class="card">
    <div class="card-body custom-card-body1">
    <a  data-bs-toggle="modal" data-bs-target="#transfer"> <center> <FontAwesomeIcon icon={faHandFist} className="icon" /> </center>
    <h5 class="card-title"><center>Transfer</center></h5> </a>
  

   
<div class="modal fade" id="transfer">
  <div class="modal-dialog">
    <div class="modal-content">

   
      <div class="modal-header">
        <h4 class="modal-title">Lots transfer to production</h4>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>

      <div class="modal-body">
      <center><h3>{Pending} Kg</h3>
       <br/>
       <button
            type="button"
            className="btn btn-success success padl"
            data-bs-dismiss="modal"            // closes the modal
            onClick={transfer}            // triggers your submit logic + navigation
          >
            Transfer
          </button>
       </center>
      </div>

      

    </div>
  </div>
</div>



   </div>
  </div>
   
  
  </div>

  

  <div class="col-sm-3">



</div>



</div>

<br></br>

<div class="row">

  <div class="col-sm-2">

  </div>

  <div class="col-sm-8">

   <div class="card">
    <div class="card-body">
   
    <LotList/>   
   <br></br>
   </div>
  </div>

  </div>

  <div class="col-sm-2">

</div>

</div>




  </main>
</div>

    
       

        </>
    )


}
export default Accepting;