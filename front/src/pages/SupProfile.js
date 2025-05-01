import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';
import man from '../img/man.jpg';
import { useParams } from 'react-router-dom';
import { useEffect } from "react";
import SupplierLotList from '../components/Supplier-LotList.js';




function SupProfile(){


  const Params = useParams();
    
  const [SupInfo,setSupInfo] = useState({FirstName:"",LastName:"",Email:"",City:"",Phone:""});

  const [Qty,setQty] = useState({ totalQtySupplierMonth:0 });
  const [All,setAll] = useState({ totalQtySupplierAll:0 });

  const SupId = Params.Code;



  useEffect(()=>{


    const loadSupplier = async() =>{

        
    const response = await axios.get(`/api/supplier/${SupId}`);

    const supplier = response.data;

    setSupInfo(supplier);

    };

    loadSupplier();

},[]);

useEffect(()=>{


  const loadMonthQty = async() =>{

      
  const response = await axios.get(`/api/supplier/qty/${SupId}`);

  const qty = response.data;

  setQty(qty);

  };

  loadMonthQty();

},[]);


useEffect(()=>{


  const loadAllQty = async() =>{

      
  const response = await axios.get(`/api/supplier/qty/all/${SupId}`);

  const all = response.data;

  setAll(all);

  };

  loadAllQty();

},[]);






    return(
        <>
  
   
<div className="app-container">
    <Header/>
  <main className="content">

  <h1 class="pagetitle">Supplier Profile</h1>
  <br></br>

<div class="row">

  <div class="col-sm-2"></div>    
  <div class="col-sm-8">

<div class="card">
<div class="card-body custom-card-body1">

  <div class="row">
  <div class="col-sm-2"><img src={`/api/supplier/${SupId}/pic`}  class="rounded-circle" width="125px" height="125px" onError={(e) => {
    // Prevent endless loop if 'man' also fails for some reason
    e.currentTarget.onerror = null; 
    e.currentTarget.src = man;
  }}></img></div>
  <div class="col-sm-4 pad2"><h3> {SupInfo.FirstName} {SupInfo.LastName}</h3></div>
  <div class="col-sm-3"> <div class="card">
    <div class="card-body">
      <h6 class="card-title"><center>This Month</center></h6>
      
      <center><h5 class="figures">{Qty.totalQtySupplierMonth.toFixed(2)} KG</h5></center>
   </div>
  </div></div>
  <div class="col-sm-3">  <div class="card">
    <div class="card-body">
      <h6 class="card-title"><center>Points</center></h6>
      
      <center><h5 class="figures">{SupInfo.Points}</h5></center>
   </div>
  </div></div>
  </div>
     
</div>
</div>
<br></br>
<div class="card">
<div class="card-body custom-card-body1">
  <div class="row">
  <div class="col">

      <p class="personinfo"><b>Supplier ID -</b> {SupInfo.Code}</p>
      <p class="personinfo"><b>NIC -</b> {SupInfo.NIC}</p>
      <p class="personinfo"><b>Email -</b> {SupInfo.Email}</p>
      <p class="personinfo"><b>Adress -</b> {SupInfo.City}</p>
      <p class="personinfo"><b>Phone Number -</b> {SupInfo.Phone}</p>
      <p class="personinfo"><b>Supplier Since -</b> {SupInfo.RegMonth} / {SupInfo.RegYear}</p>
      
      

  </div>

  </div>

  <div class="row">
  <div class="col">

      
      

  </div>

  </div>
     
</div>
</div>

  </div>

  <div class="col-sm-2"></div>
</div>
<br></br>
<div class="row">

  <div class="col-sm-2"></div>    
  <div class="col-sm-8">

      <div class="card">
    <div class="card-body">
    
    <SupplierLotList supplierId={SupId} />


      <br></br>
   </div>
  </div>

  </div>

  <div class="col-sm-2"></div>
</div>


  </main>
</div>
  

        </>
    )


}
export default SupProfile;