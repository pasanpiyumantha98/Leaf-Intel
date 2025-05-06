import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';
import man from '../img/man.jpg';
import { useParams } from 'react-router-dom';
import { useEffect } from "react";





function UserProfile(){


  const Params = useParams();
    
  const [UserInfo,setUser] = useState({FirstName:"",LastName:"",Email:"",City:"",Phone:""});

  
  const StaffId = Params.StaffId;



  useEffect(()=>{
    const loadSupplier = async() =>{
    const response = await axios.get(`/api/user/${StaffId}`);
    const user = response.data;
    setUser(user);
    };

    loadSupplier();
},[]);


    return(
        <>
  
<div className="app-container">
    <Header/>
  <main className="content">

  <h1 class="pagetitle">User Profile</h1>
  <br></br>

<div class="row">

  <div class="col-sm-2"></div>    
  <div class="col-sm-8">

<div class="card">
<div class="card-body">
  <div class="row">
  <div class="col-sm-2"><img src={man} class="rounded-circle" width="125px" height="125px"></img></div>
  <div class="col-sm-6 pad2"><h3>{UserInfo.Gender === 'Male' ? `Mr. ${UserInfo.FirstName} ${UserInfo.LastName}` : `Mrs. ${UserInfo.FirstName} ${UserInfo.LastName}`}</h3></div>
  <div class="col-sm-1"> </div>
  <div class="col-sm-3">  <div class="card">
    <div class="card-body">
      <h6 class="card-title"><center>{UserInfo.accessLevel}</center></h6>
      
      
   </div>
  </div></div>
  </div>
     
</div>
</div>

<div class="card">
<div class="card-body">
  <div class="row">
  <div class="col">

      <p class="personinfo">Staff Id - {UserInfo.StaffId}</p>
      <p class="personinfo">Gender - {UserInfo.Gender}</p>
      <p class="personinfo">Email - {UserInfo.Email}</p>
      <p class="personinfo">Adress - {UserInfo.City}</p>
      <p class="personinfo">Phone Number - {UserInfo.Phone}</p>
      <p class="personinfo">NIC - {UserInfo.NIC}</p>
      
      

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


  </main>
</div>
  

        </>
    )


}
export default UserProfile;