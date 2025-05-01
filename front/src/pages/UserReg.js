import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faInfoCircle, faEnvelope, faUser, faCog, faSuperscript, faPeopleGroup, faClover, faCancel, faHandHolding, faAd, faAdd, faGlasses, faDeleteLeft } from '@fortawesome/free-solid-svg-icons';
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";





function UserReg(){


  const [email, setemail] = useState("");
  const [fname, setfname] =useState("");
  const [lname, setlname] = useState("");
  const [city, setcity] =useState("");
  const [phone, setphone] =useState("");
  const [nic, setnic] =useState("");
  const [uname, setuname] =useState("");
  const [gender, setgender] =useState("Male");
  const [accesslevel, setaccess] =useState("Manager/Owner");
  

  
  const navigate = useNavigate();
  
  
  async function Submitform()
  {
  
  if(fname===""|lname===""|email===""|city===""|phone===""|nic===""|uname==="")
  {
  
    toast.error("UNeed to fill all blanks!");
  
  } else
  {
  
  const pass ="123"  

  const response = await axios.post(`/users-registration/go`, {fname:fname,lname:lname,city:city,email:email,phone:phone,nic:nic, uName:uname, gender:gender,accesslevel:accesslevel,pass:pass});
  
  
  const stat=response.data;
  
  if(stat==="OK")
  {

  
    toast.success("Successfully Registered! Password - 123");
    navigate("/users");


  } else if(stat==="UserExist")
  {
    toast.error("User Already Registered!");
  } else {
    toast.error("Username Already Exsist!");
  }
  
  }
  
  
  };

  const handleGenderChange = (event) => {
    setgender(event.target.value);
  };

  const handleAccessChange = (event) => {
    setaccess(event.target.value);
  };





    return(
        <>

    
<ToastContainer />
<div className="app-container">
    <Header/>
  <main className="content">


    <h1 class="pagetitle">User Registration</h1>

<br></br>

<div class="row">

  <div class="col-sm-2">

 

  </div>

  <div class="col-sm-8">

   <div class="card">
    <div class="card-body">
  
<div class="row">

<div class="col-sm-1">
</div>


<div class="col-sm-4">
<label ><b>First Name - </b></label> 
<input type="text" placeholder="Enter first name" name="fname" value={fname} onChange={r=>setfname(r.target.value)} required></input>
<br/> <br/>
<label><b>Last Name - </b></label> 
<input type="text" placeholder="Enter last name" name="lname" value={lname} onChange={g=>setlname(g.target.value)} required></input>
<br/> <br/>
<label ><b>Adress- </b></label> 
<input type="text" placeholder="Enter City" name="city" value={city} onChange={t=>setcity(t.target.value)} required></input>
<br/> <br/>
<label ><b>Gender- </b></label> 
<select
    id="gender"
    name="gender"
    value={gender}
    onChange={handleGenderChange}
    className="form-select"
  >
    <option value="male">Male</option>
    <option value="female">Female</option>
    <option value="other">Other</option>
  </select>


<br/> <br/>
<label ><b>Access Level- </b></label> 
<select
    id="accesslevel"
    name="accesslevel"
    value={accesslevel}
    onChange={handleAccessChange}
    className="form-select"
  >
    <option value="Manager/Owner">Manager/Owner</option>
    <option value="Factory Officer">Factory Officer</option>
    <option value="Billing Clerk">Billing Clerk</option>
  </select>
<br/> <br/>


</div>

<div class="col-sm-1">
</div>

<div class="col-sm-4">
<label ><b>Email ID- </b></label>  
<input type="text" placeholder="Enter Email" name="email" value={email} onChange={q=>setemail(q.target.value)} required></input>
<br/> <br/>
<label ><b>Phone- </b></label>  
<input type="text" placeholder="Enter Phone" name="phone" value={phone} onChange={e=>setphone(e.target.value)} required></input>
<br/> <br/>
<label ><b>NIC- </b></label>  
<input type="text" placeholder="Enter NIC" name="nic" value={nic} onChange={i=>setnic(i.target.value)} required></input>
<br/> <br/>
<label ><b>Username- </b></label> 
<input type="text" placeholder="Enter User Name" name="city" value={uname} onChange={t=>setuname(t.target.value)} required></input>
<br/> <br/>
<label ><b>Password- </b></label> 
<input type="text" placeholder="Enter User Name" name="city" value="123" readOnly></input>
<br/> <br/>


  



</div>

<div class="col-sm-2">
</div>

</div>

<center><button onClick={Submitform} class="success">Register</button></center>

<br/> <br/>
      
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
export default UserReg;