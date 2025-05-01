import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState,useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';
import success from '../img/success.png';
import { useParams } from 'react-router-dom';





function AdvanceSuccess(){

  const Params = useParams();
  const TransId = Params.TransId;
  const [Trans, setTrans] = useState("");
  const [Supp, setSupInfo] = useState("");

 
  useEffect(()=>{


    const loadSupplier = async() =>{

        
    const response = await axios.get(`/api/transaction/${TransId}`);

    const trans = response.data;

    setTrans(trans);

    const response2 = await axios.get(`/api/supplier/${response.data.SuppId}`);
    
    const supplier = response2.data;

    setSupInfo(supplier);

    };

    loadSupplier();

},[]);


 
    
    const navigate = useNavigate();


  
    return(
        <>
  
  <div className="app-container">
    <Header/>
  <main className="content">


  


<div class="row">

  <div class="col-sm-3">

  </div>

  <div class="col-sm-6">
    <br></br><br></br>
  <div class="card">
  <div class="card-body">
  
  <center><img src={success}></img>
  <h2 class="conftitle">{Trans.Amount} LKR</h2>
  <h3 class="conftitle">Advance Successfully Issued for {Supp.FirstName} ! </h3></center>
  <br></br>
  <p class="conftext"><b>Transaction ID -</b> {Trans.TransId}</p>
  <p class="conftext"><b>Supplier ID -</b> {Trans.SuppId}</p>
  <p class="conftext"><b>Staff -</b>{Trans.Staff} </p>
  <p class="conftext"><b>Date -</b> {Trans.Date}/{Trans.Month}/{Trans.Year}</p>

  <center><a href={"/billing"} class="success padl nounder">Done</a></center>


  </div>
  </div>

  </div>

  <div class="col-sm-3">



   </div>
  </div>
  
  






  </main>
</div>

    
       

        </>
    )


}
export default AdvanceSuccess;