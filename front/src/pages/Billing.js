import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js'
import Header from '../components/Header';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faMoneyBill1,faCheck, faDeleteLeft, faMoneyBillTransfer } from '@fortawesome/free-solid-svg-icons';
import { useEffect } from 'react';
import React from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import BillingChart from '../components/BillingChart';
import PaymentList from '../components/Payment-List';






function Billing(){

  const [supid, setsupid] = useState("");
  const [SuppId, setSuppID] = useState("");
  const [ammount, setAmount] =useState("");
  const [TID, setTID] =useState("");
  const [Rate, setRate] =useState("Loading..");
  const [MaxAdvance, setMaxAdvance] =useState("Loading..");
  const [LastPayment, setLastPayment] =useState("Loading..");
  const [ThisPayments, setThisPayments] =useState("Loading..");

  
  const navigate = useNavigate();

  useEffect(()=>{

    const loadTopSup = async()=>{


      const response = await axios.get(`/api/settings/rate`);

      const rate = response.data;

      setRate(rate.Value);


      const response3 = await axios.get(`/api/settings/maxadvance`);

      const mxadvnce = response3.data;
  
      setMaxAdvance(mxadvnce.Value);

      
      const response4 = await axios.get(`/api/payments/last`);

      const lpayment = response4.data;
  
      setLastPayment(lpayment);


      const response5 = await axios.get(`/api/payments/thismonth`);

      const thisMonth = response5.data;
  
      setThisPayments(thisMonth);

      


    };

    loadTopSup();


  },[]);


  async function checkADVNCE()

  {


    const user = JSON.parse(localStorage.getItem('user'));  

    const un = user.UserName;

      const response = await axios.get(`/api/advance/elg/${SuppId}/${ammount}/${un}`)

      const TransId = response.data;

      if(TransId==="CantGive-notavlble")
      {
        toast.error("Requested amount exceed the availble amount!");
        

      } else if (TransId==="NoSupplier"){
        toast.error("Supplier ID incorrect!");
      } else if (TransId==="Higher")
      {
        toast.error("Requested amount is higher!");
      } else
      {
        navigate('/advance-success/'+TransId);
      }
      

  }

  async function validateViewTransaction()
  {

    if(TID=="")
    {
      toast.error("Transaction ID is needed!");

    } else {

      const response = await axios.get(`/api/transaction/${TID}`)

      const Transaction = response.data;

      if(Transaction==="TransactionNotFound")
      {
        toast.error("Transaction ID is not correct!");
        
      }  else
      {
        navigate('/transaction/'+TID);
      }

    }
      

  }

  async function validatePayment()
  {

    if(SuppId=="")
    {
      toast.error("Supplier ID is needed!");

    } else {

      const response = await axios.get(`/api/supplier/payment/${SuppId}`) 

      const Payable = response.data;

      if(Payable==="SupplierNotFound")
      {
        toast.error("Supplier ID is incorrect!");
        
      }else if(Payable===0)
      {
        toast.error("No pending balance to pay!");
      }else if(Payable==="AlreadyIssued")
      {
          toast.error("Payment already issued for last month!");
      }else  
      {
        navigate('/payment-final', {state: { Payable: Payable, SuppId: SuppId }});
      }

    }
      

  }

    return(
        <>

    
<ToastContainer />
<div className="app-container">
    <Header/>
  <main className="content">


    <h1 class="pagetitle">Billing & Payments</h1>

    <div class="container">
  <div class="row">
    
    <div class="col-sm-5">
      <div class="card">
        <div class="card-body">
         
          <div class="row">
            <div class="col-sm-6 card-billing-set1" >
              <h6>Buying Rate</h6>
              <h4>{Rate} LKR/Kg</h4>
            </div>
            <div class="col-sm-6 card-billing-set2" >
            <h6>Supplier Advance Limit</h6>
            <h4>{MaxAdvance} LKR</h4>
            </div>
          </div>
          <div class="row">
            <div class="col-sm-6 card-billing-set3" >
            <h6>Last Transaction</h6>
            <h4>{LastPayment.Amount} LKR</h4><h6> ({LastPayment.Type})</h6>
            </div>
            <div class="col-sm-6 card-billing-set4" >
            <h6>This Month Payments</h6>
            <h4>{ThisPayments} LKR</h4>
            </div>
          </div>
        </div>
      </div>

<br></br>

      <div class="row">

  <div >

 

  </div>

  <div class="col-sm-4">

<div class="card">
 <div class="card-body custom-card-body1">
 <a  data-bs-toggle="modal" data-bs-target="#Pay"> <center> <FontAwesomeIcon icon={faMoneyBill1} className="icon" /> </center>
 <h5 class="card-title"><center>Pay <br/>Suppliers</center></h5> </a>



<div class="modal fade" id="Pay">
<div class="modal-dialog">
 <div class="modal-content">


   <div class="modal-header">
     <h4 class="modal-title">Supplier Payments</h4>
     <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
   </div>

   <div class="modal-body">
   
   <label>Supplier ID :&ensp; &ensp;</label>
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

        


    <br/><a onClick={validatePayment} data-bs-dismiss="modal" class="success padl nounder">Finalize</a>
   </div>

   

 </div>
</div>
</div>



</div>
</div>

</div>


  <div class="col-sm-4">

   <div class="card">
    <div class="card-body custom-card-body1">
    <a  data-bs-toggle="modal" data-bs-target="#search"> <center> <FontAwesomeIcon icon={faMoneyBillTransfer} className="icon" /> </center>
    <h5 class="card-title"><center>Issue Advance</center></h5> </a>
  

   
<div class="modal fade" id="search">
  <div class="modal-dialog">
    <div class="modal-content">

   
      <div class="modal-header">
        <h4 class="modal-title">Issueing Advances</h4>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>

      <div class="modal-body">
      <label>Supplier ID :&ensp; &ensp;</label>
        <input
          className="inputfsmall"
          type="text"
          placeholder="Supplier ID"
          name="ID"
          value={SuppId}
          onChange={(q) => setSuppID(q.target.value)}
          required
        />
      <br/>
      <label>Amount :&ensp; &ensp;&ensp; &ensp;</label>
        <input
          className="inputfsmall"
          type="text"
          placeholder="Amount"
          name="ID"
          value={ammount}
          onChange={(q) => setAmount(q.target.value)}
          required
        />

       <center>
       <br/><a onClick={checkADVNCE} class="success padl nounder"  data-bs-dismiss="modal">Proceed</a></center>
      </div>

      

    </div>
  </div>
</div>



   </div>
  </div>
  
  </div>

  <div class="col-sm-4">

   <div class="card">
<div class="card-body custom-card-body1">
   
     <a  data-bs-toggle="modal" data-bs-target="#delete"><center> <FontAwesomeIcon icon={faCheck} className="icon" /> </center>
    <h5 class="card-title"><center>View Transaction</center></h5> </a>


    <div class="modal fade" id="delete">
  <div class="modal-dialog">
    <div class="modal-content">

    <div class="modal-header">
        <h4 class="modal-title">View Transaction</h4>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>
      

      <div class="modal-body">
       
      <label>Transaction ID :&ensp; &ensp;&ensp; &ensp;</label>
        <input
          className="inputfsmall"
          type="text"
          placeholder="Transaction ID"
          name="ID"
          value={TID}
          onChange={(q) => setTID(q.target.value)}
          required
        />

       <br/>
       <center><a onClick={validateViewTransaction} class="success padl nounder" data-bs-dismiss="modal">View</a></center>
      </div>

      

    </div>
  </div>
</div>



   </div>
  </div>
  
  </div>

  

  <div class="col-sm-1">



</div>



</div>
    </div>
    <div class="col-sm-5">
      <div class="card">
        <div class="card-body">
         
        <BillingChart/>
        </div>
      </div>
    </div>
   
  </div>
</div>

<br/>



<br></br>
<br></br>

<div class="row">

 

  <div class="col-sm-10">

   <div class="card">
    <div class="card-body">

        <PaymentList/>
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
export default Billing;