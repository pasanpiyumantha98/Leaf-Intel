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





function SupReg(){


  const [email, setemail] = useState("");
  const [fname, setfname] =useState("");
  const [lname, setlname] = useState("");
  const [city, setcity] =useState("");
  const [phone, setphone] =useState("");
  const [nic, setnic] =useState("");
  const [pic, setProPic] = useState(null);

  
  const navigate = useNavigate();

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setProPic(file);
  };
  
  
  async function Submitform()
  {
  
  if(fname===""|lname===""|email===""|city===""|phone===""|nic==="")
  {
  
       toast.error("Need to fill all blanks!");
  
  } else
  {

    const formData = new FormData();
    formData.append('fname', fname);
    formData.append('lname', lname);
    formData.append('city', city);
    formData.append('email', email);
    formData.append('phone', phone);
    formData.append('nic', nic);
    formData.append('pic', pic);

    const response = await axios.post(
      '/suppliers-registration/go', 
      formData, 
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    );

  
  const stat=response.data;
  
  if(stat==="OK")
  {

  
    toast.success("Successfully Registered!");
    navigate("/suppliers");


  } else 
  {
       toast.error("Supplier Already Registered!");
  }
  
  }
  
  
  };




    return(
        <>

<ToastContainer />
   
<div className="app-container">
    <Header/>
  <main className="content">


    <h1 class="pagetitle">Suppliers Registration</h1>

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

<label ><b>Profile Picture- </b></label> 
<input type="file"   onChange={handleFileChange} required></input>
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


  



</div>

<div class="col-sm-2">
</div>

</div>

<center><button onClick={Submitform} class="success">Register</button></center>


      
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
export default SupReg;