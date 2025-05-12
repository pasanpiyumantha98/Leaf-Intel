import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { Typography } from '@mui/material';


function Login()
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
    <br/><br/><br/><br/><br/><br/>
    


    <div class="container-fluid mt-3">
  
  <div class="row">
    <div class="col-sm-4 p-3  text-white"></div>
    <div class="col-sm-4 p-3 text-white">

    <div class="bg-primary-transparent text-black rounded">
    <br/>
  <center> <Typography variant="h4" component="h3">
                    Staff Login
                  </Typography></center>
  <br/>
  
 <center> 


 <div class="form-group row">
  <div class="col-xs-2">

    <input class="form-control" type="text" placeholder="UserName" name="uname" value={uname} onChange={q=>setuname(q.target.value)} required></input>
  </div>
  <div class="col-xs-3">
   
    <input class="form-control" type="password" placeholder="Password" name="pass" value={pass} onChange={e=>setpass(e.target.value)} required></input>
  </div>

</div>

  <button class="success" onClick={submitlogin}>Login</button>
  <br/>
  <br/>
 <Link to="/signup"> <p>Having issues?</p></Link>
 <br/>

 

  </center>

  </div>


    </div>
    <div class="col-sm-4 p-3  text-white">
   
    </div>
  </div>
</div>

<br/>
<br/>
 <br/>
 <br/>
 <br/>
 <br/> 

    </body>
    </>


);


}
export default Login;