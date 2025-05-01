import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import vector from '../img/login.png';
import logo from '../img/logo1.png';


function Login1()
{


    const [uname, setuname] = useState("");
    const [pass, setpass] =useState("");
    
    const navigate = useNavigate();


    async function submitlogin()
    {

        const person = await axios.post(`/login/go`,{uname:uname,pass:pass})

        if(person.data.NIC)
        {
          localStorage.setItem('token', person.data.token);
          localStorage.setItem('user', JSON.stringify(person.data));
          
          
          navigate('/dashboard');
        } else{
          toast.error("Wrong credentials!");
        }
        

    }




return(

  

    <>

<ToastContainer
/>
    <body className='log'>
    <br/><br/>
   
    

   <center> <img src={logo} width="100x" height="100px"></img></center>
    <div class="container-fluid mt-1">
  
  <div class="row">
    <div class="col-sm-3 p-1  text-white"></div>
    <div class="col-sm-6 p-1 text-white">

    <div class="bg-primary-transparent text-black rounded">
    <br/> <br/>
    <div class="row">
  <div class="col-sm-6">

  <center><img src={vector} width="300px" height="360px"></img></center>

  </div>
  
 
  <div class="col-sm-6">

   
    <br/>
    <center>
    
       <h3 class="custom-heading"><b>Hello<br/> Welcome back!</b></h3></center> 
  <br/>
  
 <center> 


 <div class="form-group row">
  <div class="col-xs-2">

    <input class="form-control" type="text" placeholder="Username" name="uname" value={uname} onChange={q=>setuname(q.target.value)} required></input>
  </div>
  <div class="col-xs-3">
   
    <input class="form-control" type="password" placeholder="Password" name="pass" value={pass} onChange={e=>setpass(e.target.value)} required></input>
  </div>

</div>

  <button class="success" onClick={submitlogin}>Login</button>
  <br/>
  <br/>

 <br/>

 

  </center>

 
 </div>

  </div>


    </div>
    <div class="col-sm-3 p-1  text-white">
    </div>
   
    </div>
  </div>
</div>

<br/>
<br/>
 <br/>
 <br/>



    </body>
    </>


);


}
export default Login1;