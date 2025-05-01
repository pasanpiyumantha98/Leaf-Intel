import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState,useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';
import payment from '../img/payment.png';
import { useParams, useLocation } from 'react-router-dom';


function PaymentFinal(){

  const Params = useParams();

  const { state } = useLocation();
  const { Payable, SuppId } = state || {};
  const qty =3499;
  
  const [Supplier, setSupplier] = useState("");
  const [Trans, setTrans] = useState("");
  const [Rate, setRate] = useState("");
  const [User, setUser] = useState("");
  const [Month, setMonth] = useState("");
  const [Year, setYear] = useState("");

  const navigate = useNavigate();

  const user = JSON.parse(localStorage.getItem('user'));  

  const un = user.UserName;

  useEffect(()=>{


    const loadSupplier = async() =>{

        
    const response = await axios.get(`/api/supplier/${SuppId}`);

    const Supplier = response.data;

    setSupplier(Supplier);

    const response2 = await axios.get(`/api/settings/rate`);

    const rate = response2.data;

    setRate(rate.Value);

    const user = JSON.parse(localStorage.getItem('user'));  

    setUser(user.UserName);

    };

    loadSupplier();

},[]);

useEffect(()=>{


  const dates = async() =>{

    const currentDate = new Date();
    let y=currentDate.getFullYear();
    setYear(y);
    const m = currentDate.getMonth() -1;

    if(m>=0)
    {

  if (m === 0) {
  setMonth("January");
} else if (m === 1) {
  setMonth("February");
} else if (m === 2) {
  setMonth("March");
} else if (m === 3) {
  setMonth("April");
} else if (m === 4) {
  setMonth("May");
} else if (m === 5) {
  setMonth("June");
} else if (m === 6) {
  setMonth("July");
} else if (m === 7) {
  setMonth("August");
} else if (m === 8) {
  setMonth("September");
} else if (m === 9) {
  setMonth("October");
} else if (m === 10) {
  setMonth("November");
} else if (m === 11) {
  setMonth("December");
} else {
  setMonth("Invalid Month"); 
}

    } else 
    {
      setMonth("December");
      y = y-1
      setYear(y);
    }
    


  };

  dates();

},[]);



  const [modalShow, setModalShow] = useState(false);
 

  const MakePayment = async () => {
   
      const response = await axios.post(`/api/supplier/makepay`, {
        SuppId,
        Payable,
        Month,
        Year,
        un
      });

    const TransId = response.data;

    const Trans = await axios.get(`/api/transaction/${TransId}`);

    const Trans1 = Trans.data;

    setTrans(Trans1);
    


      setModalShow(true);
    } 

 
    
    


  
    return(
        <>
  
  <div className="app-container">
    <Header/>
  <main className="content">

 {/* Bootstrap Modal */}
 {modalShow && (
        <div className="modal fade show" tabIndex="-1" style={{ display: 'block' }} aria-hidden="false">
          <div className="modal-dialog">
            <div className="modal-content">
              <div className="modal-header">
                <h5 className="modal-title">Payment Successfuly Recorded!</h5>
                <button
                  type="button"
                  className="btn-close"
                  data-bs-dismiss="modal"
                  aria-label="Close"
                  onClick={() => {
                    setModalShow(false);
                    navigate('/billing'); // Navigate to /billing after closing the modal
                  }}
                ></button>
              </div>
              <div className="modal-body">
                <p>Transaction ID: {Trans.TransId}</p>
                <p>Supplier ID: {Trans.SuppId}</p>
                <p>Payment For: {Month}/{Year}</p>
                <p>Amount : <b>{Trans.Amount.toFixed(2)} LKR</b></p>
                <p>Quantity : {qty} KG</p>
                <p>Rate : {Rate} LKR</p>
                <p>Payment Date : {Trans.Date}/{Trans.Month}</p>
                <p>Staff: {Trans.Staff}</p>
                <p>The payment has been successfully processed.</p>
              </div>
              <div className="modal-footer">
              <button
                  type="button"
                  className="btn btn-secondary"
                  data-bs-dismiss="modal"
                  onClick={() => {
                    setModalShow(false);
                    navigate('/billing'); // Navigate to /billing after closing the modal
                  }}
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
  


<div class="row">

  <div class="col-sm-3">

  </div>

  <div class="col-sm-6">
    <br></br> <br></br>
  <div class="card">
  <div class="card-body">
  
  <center><img src={payment}></img>
  <h2 class="conftitle">{Payable.toFixed(2)} LKR <br/>For {Month}, {Year}</h2></center>
  <br></br>
  
 
  <p class="conftext"><b>Supplier Name -</b> {Supplier.FirstName} {Supplier.LastName}</p>
  <p class="conftext"><b>Supplier ID -</b> {SuppId} </p>
  <p class="conftext"><b>Staff -</b> {User} </p>
  

  <center><a onClick={MakePayment} class="success padl nounder">Pay</a></center>
  


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
export default PaymentFinal;