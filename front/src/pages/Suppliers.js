import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js'
import Header from '../components/Header';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPlus,faAdd, faGlasses, faDeleteLeft } from '@fortawesome/free-solid-svg-icons';
import { useEffect } from 'react';
import SupplierList from '../components/Supplier-List';




function Suppliers(){

  const [supid, setsupid] = useState("");

  const [SupInfo,setSupInfo] = useState([]);




  useEffect(()=>{


    const loadSupplier = async() =>{

        
    const response = await axios.get(`/api/suppliers`);

    const supplier = response.data;

    setSupInfo(supplier);

    };

    loadSupplier();

},[]);





    return(
        <>

    
   
<div className="app-container">
    <Header/>
  <main className="content">


    <h1 class="pagetitle">Suppliers</h1>


<div class="row">

  <div class="col-sm-3">

 

  </div>

  <div class="col-sm-2">

   <div class="card">
    <div class="card-body custom-card-body1">
  
    <a href="/supplier-registration" class="nounder"> <center> <FontAwesomeIcon icon={faPlus} style={{ color: 'black' }} className="icon" /> </center>
      <h5 class="card-title" style={{ color: 'black' }} ><center>Create</center></h5></a>
   </div>
  </div>

  </div>

  <div class="col-sm-2">

   <div class="card">
    <div class="card-body custom-card-body1">
    <a  data-bs-toggle="modal" data-bs-target="#search"> <center> <FontAwesomeIcon icon={faGlasses} className="icon" /> </center>
    <h5 class="card-title"><center>View</center></h5> </a>
  

   
<div class="modal fade" id="search">
  <div class="modal-dialog">
    <div class="modal-content">

   
      <div class="modal-header">
        <h4 class="modal-title">Supplier Search</h4>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>

      <div class="modal-body">
       <center><input type="text" placeholder="Enter Suppier ID" name="ID" value={supid} onChange={q=>setsupid(q.target.value)} required></input>
       <br/><a href={"supplier-profile/"+supid} class="success padl nounder">Find</a></center>
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
     <a  data-bs-toggle="modal" data-bs-target="#delete"><center> <FontAwesomeIcon icon={faDeleteLeft} className="icon" /> </center>
    <h5 class="card-title"><center>Remove</center></h5> </a>


    <div class="modal fade" id="delete">
  <div class="modal-dialog">
    <div class="modal-content">

   
      <div class="modal-header">
        <h4 class="modal-title">Supplier Delete</h4>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>

      <div class="modal-body">
       <center><input type="text" placeholder="Enter Suppier ID" name="ID" value={supid} onChange={q=>setsupid(q.target.value)} required></input>
       <br/><a href={"supplier-delete/"+supid} class="success padl nounder">Remove</a></center>
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

        <SupplierList/>
   
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
export default Suppliers;