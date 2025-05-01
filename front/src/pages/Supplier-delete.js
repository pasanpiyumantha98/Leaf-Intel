import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';
import man from '../img/man.jpg';
import { useParams } from 'react-router-dom';
import { useEffect } from "react";




function SupDelete(){


  const Params = useParams();
    

  const SupId = Params.id;



  useEffect(()=>{


    const supplierdelete = async() =>{

        
    await axios.get(`/api/supplier-delete/${SupId}`);

    

   

    };

    supplierdelete();

    

},[]);



    return(
        <>
  
   
<div className="app-container">
    <Header/>
  <main className="content">
  <br></br><br></br>
  <center><h1 >Supplier Deleted successfully!</h1>
  <br></br>
  <a href="/suppliers" class="success nounder">Go Back</a> </center>



  </main>
</div>
  

        </>
    )


}
export default SupDelete;