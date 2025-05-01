import '../App.css';
import {useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js'
import Header from '../components/Header';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faInfoCircle, faEnvelope, faUser, faCog, faSuperscript, faPeopleGroup, faClover, faCancel, faHandHolding, faAd, faAdd, faGlasses, faDeleteLeft } from '@fortawesome/free-solid-svg-icons';
import UsersList from '../components/Users-List';
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";





function Users(){

  const [userid, setuserid] = useState("");

  const user = JSON.parse(localStorage.getItem('user'));  

  const navigate = useNavigate();

  const sid = user.StaffId;

  const userDelete = async () => {
    if (parseInt(userid) === sid) {
      toast.error("User cannot be removed!");
      navigate('/users');
    } else {
      try {
        await axios.get(`/api/user-delete/${userid}`);
        toast.success("User removed!");

      } catch (error) {
        console.error("Error deleting user:", error);
        toast.error("Failed to remove user.");
      }
    }
  };
  




    return(
        <>

<ToastContainer />
   
<div className="app-container">
    <Header/>
  <main className="content">


    <h1 class="pagetitle">Users</h1>


<div class="row">

  <div class="col-sm-3">

 

  </div>

  <div class="col-sm-2">

  

  </div>

  <div class="col-sm-2">

   <div class="card">
    <div class="card-body">
    <a href="/users/registration" class="nounder"> <center> <FontAwesomeIcon icon={faAdd} className="icon" /> </center>
      <h5 class="card-title"><center>Add User</center></h5></a>
   </div>
  </div>
  
  </div>

  <div class="col-sm-2">

   <div class="card">
    <div class="card-body">
     <a  data-bs-toggle="modal" data-bs-target="#delete"><center> <FontAwesomeIcon icon={faDeleteLeft} className="icon" /> </center>
    <h5 class="card-title"><center>Remove User</center></h5> </a>


    <div class="modal fade" id="delete">
  <div class="modal-dialog">
    <div class="modal-content">

   
      <div class="modal-header">
        <h4 class="modal-title"> Remove User</h4>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>

      <div class="modal-body">
       <center><input type="text" placeholder="Enter Staff ID" name="ID" value={userid} onChange={q=>setuserid(q.target.value)} required/>
       <br/><button onClick={userDelete} class="success padl nounder">Remove</button></center>
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

        <UsersList/>
   
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
export default Users;