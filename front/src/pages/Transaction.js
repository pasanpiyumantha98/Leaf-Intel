import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState,useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';
import success from '../img/success.png';
import { useParams } from 'react-router-dom';
import trans from '../img/transaction.png';





function Transaction(){

  const Params = useParams();
  const TransId = Params.TransId;
  const [Trans, setTrans] = useState("");

 
  useEffect(()=>{


    const loadSupplier = async() =>{

        
    const response = await axios.get(`/api/transaction/${TransId}`);

    const trans = response.data;

    setTrans(trans);

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
    <br></br>
  <div class="card">
  <div class="card-body">
  
  <center><img src={trans}></img>
  <h2 class="conftitle"> Supplier Transactions</h2></center>
  <br></br>
  <p class="conftext"><b>Transaction ID -</b> {Trans.TransId}</p>
  <p class="conftext"><b>Supplier ID -</b>  {Trans.SuppId}</p>
  <p class="conftext"><b>Type -</b>  {Trans.Amount}</p>
  <p class="conftext"><b>Amount - </b> {Trans.Type}</p>
  <p class="conftext"><b>Staff -</b> {Trans.Staff} </p>
  <p class="conftext"><b>Date -</b>  {Trans.Date}/{Trans.Month}/{Trans.Year}</p>

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
export default Transaction;