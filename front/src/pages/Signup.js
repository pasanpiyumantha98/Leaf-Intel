import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState } from 'react';
import axios from "axios";

function SignupPage()
{

const [email, setemail] = useState("");
const [fname, setfname] =useState("");
const [lname, setlname] = useState("");
const [pass, setpass] =useState("");
const [city, setcity] =useState("");
const [gender, setgender] =useState("");

const navigate = useNavigate();


async function Submitform()
{

if(fname===""|lname===""|email===""|pass===""|city===""|gender==="")
{

    alert("You need to fill all blanks!");

} else  if(pass.length<7)
{
    alert("Your Password needs to have more than 6 digits")

} else
{
const response = await axios.post(`/signup/go`, {fname:fname,lname:lname,city:city,gender:gender,email:email,pass:pass});


const stat=response.data;

if(stat==="OK")
{
navigate('/login'); 
} else 
{
    alert("User already registered.");
}

}


};





    return(
        <>
        
        <br/><br/><br/><br/><br/><br/>
    <center><h2>Signup Page</h2></center>
    <br/><br/>

    

    <center> <label ><b>First Name - </b></label>
    <input type="text" placeholder="Enter first name" name="fname" value={fname} onChange={r=>setfname(r.target.value)} required></input>
    <br/>  <br/>
     <label><b>Last Name - </b></label>
    <input type="text" placeholder="Enter last name" name="lname" value={lname} onChange={g=>setlname(g.target.value)} required></input>
    <br/>  <br/>


    <br/>  <br/>

    <label ><b>City- </b></label>
    <input type="text" placeholder="Enter City" name="city" value={city} onChange={t=>setcity(t.target.value)} required></input>
    <br/>  <br/>
   <label ><b>Email ID- </b></label>
    <input type="text" placeholder="Enter Email" name="uname" value={email} onChange={q=>setemail(q.target.value)} required></input>
    <br/>  <br/>
    <label ><b>Password - </b></label>
    <input type="password" placeholder="Enter Password" name="pass" value={pass} onChange={e=>setpass(e.target.value)} required></input>
    <br/>  <br/>
    <button onClick={Submitform} class="success">Signup</button>
    
    <br/>
   <Link to="/login"> <p>I am already a member!</p> </Link>



    </center>
        
        </>

    );


}
export default SignupPage;