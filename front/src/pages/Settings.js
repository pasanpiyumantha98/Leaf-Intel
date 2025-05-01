import '../App.css';
import { Link,useNavigate } from 'react-router-dom';
import { useState,useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";





function Settings(){

  const [isBlocked, setIsBlocked] = useState(true);
  const [uname, setuname] = useState("Loading..");
  const [email, setEmail] = useState("Loading..");
  const [password, setPassword] = useState("");
  const [address, setAddress] = useState("Loading..");
  const [phone, setPhone] = useState("Loading..");
  const [rate, setRate] = useState("Loading..");
  const [goal, setGoal] = useState("Loading..");
  const [ygoal, setyGoal] = useState("Loading..");
  const [dgoal, setdGoal] = useState("Loading..");
  const [accessL, setAccessL] = useState("Loading..");
  const [staffId, setStaffId] = useState("Loading..");
  const [maxAdvance, setMaxAdvance] = useState("Loading..");
  const [notice, setNotice] = useState("Loading..");

  const [user1, setUsInfo] = useState({});

 

  useEffect(()=>{

    const loadUsers = async() =>{
      
    const user = JSON.parse(localStorage.getItem('user'));  

    const sid = user.StaffId;
    
    const response = await axios.get(`/api/user/${sid}`);
    
    const user2 = response.data;
    
    setUsInfo(user2);

    setuname(user2.UserName);
    setAddress(user2.City);
    setPhone(user2.Phone);
    setEmail(user2.Email);
    setAccessL(user2.accessLevel);
    setStaffId(user2.StaffId);  

    const response1 = await axios.get(`/api/settings/rate`);

    const set = response1.data;

    setRate(set.Value);

    const response2 = await axios.get(`/api/settings/maxadvance`);

    const set1 = response2.data;

    setMaxAdvance(set1.Value);


    const response3 = await axios.get(`/api/settings/notice`);

    const set2 = response3.data;

    setNotice(set2.Value);

    const response4 = await axios.get(`/api/settings/mgoal`);

    const set3 = response4.data;

    setGoal(set3.Value);

    const response5 = await axios.get(`/api/settings/ygoal`);

    const set4 = response5.data;

    setyGoal(set4.Value);

    const response6 = await axios.get(`/api/settings/dgoal`);

    const set5 = response6.data;

    setdGoal(set5.Value);





    };

    loadUsers();

},[]);

  function edit(){
    setIsBlocked(false);
  }

  async function update(){

   
      const response = await axios.post(`/settings/update`, {uname:uname,email:email,address:address,phone:phone,password:password,rate:rate, maxadvance:maxAdvance, notice:notice, staffid:staffId,goal:goal,ygoal:ygoal,dgoal:dgoal});
    

    setIsBlocked(true);
    
    toast.success("Settings Saved!");
    //alert("Success!");
 
  }

  

    return(
        <>

<ToastContainer />
  
   
<div className="app-container">
    <Header/>
  <main className="content">

  <h1 class="pagetitle">Settings</h1>

<div class="row">

<div class="col-sm-6">

<div class="card">
<div class="card-body">
  <div class="row">
  <div class="col"><h5>Account Settings</h5></div>
  </div>
     
</div>
</div>

<div class="card">
<div class="card-body">
<div className="container">
  {/* Username */}
  <div className="row mb-3">
    <label htmlFor="username" className="col-sm-2 col-form-label">
      Username
    </label>
    <div className="col-sm-10">
      <input
        className="form-control"
        type="text"
        id="username"
        
        name="uname"
        value={uname}
        onChange={e => setuname(e.target.value)}
        required
        disabled={isBlocked}
      />
    </div>
  </div>

  {/* Email */}
  <div className="row mb-3">
    <label htmlFor="email" className="col-sm-2 col-form-label">
      Email
    </label>
    <div className="col-sm-10">
      <input
        className="form-control"
        type="text"
        id="email"
        placeholder="Email"
        name="email"
        value={email}
        onChange={e => setEmail(e.target.value)}
        required
        disabled={isBlocked}
      />
    </div>
  </div>

  {/* Password */}
  <div className="row mb-3">
    <label htmlFor="password" className="col-sm-2 col-form-label">
      Password
    </label>
    <div className="col-sm-10">
      <input
        className="form-control"
        type="password"
        id="password"
        placeholder="Password"
        name="password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        required
        disabled={isBlocked}
      />
    </div>
  </div>

  {/* Address */}
  <div className="row mb-3">
    <label htmlFor="address" className="col-sm-2 col-form-label">
      Address
    </label>
    <div className="col-sm-10">
      <input
        className="form-control"
        type="text"
        id="address"
        placeholder="Address"
        name="address"
        value={address}
        onChange={e => setAddress(e.target.value)}
        required
        disabled={isBlocked}
      />
    </div>
  </div>

  {/* Phone */}
  <div className="row mb-3">
    <label htmlFor="phone" className="col-sm-2 col-form-label">
      Phone
    </label>
    <div className="col-sm-10">
      <input
        className="form-control"
        type="text"
        id="phone"
        placeholder="Phone Number"
        name="phone"
        value={phone}
        onChange={e => setPhone(e.target.value)}
        required
        disabled={isBlocked}
      />
    </div>

   
    
  </div>
</div>

     
</div>
</div>

  </div>

  <div class="col-sm-6">

  {accessL === "Manager" && (
<div class="card">
<div class="card-body">
  <div class="row">
 
  <div class="col"><h5>System Settings</h5></div>
  </div>
     
</div>
</div>
  )}


{accessL === "Manager" && (
<div class="card">
<div class="card-body">
<div className="container">

  <>
    {/* Paying Rate */}
    <div className="row mb-3">
      <label htmlFor="username" className="col-sm-4 col-form-label">
        Paying Rate
      </label>
      <div className="col-sm-8">
        <input
          className="form-control"
          type="text"
          id="rate"
          name="uname"
          value={rate}
          onChange={e => setRate(e.target.value)}
          required
          disabled={isBlocked}
        />
      </div>
    </div>

    {/* Maximum advance amount */}
    <div className="row mb-3">
      <label htmlFor="email" className="col-sm-4 col-form-label">
        Maximum Advance Amount
      </label>
      <div className="col-sm-8">
        <input
          className="form-control"
          type="text"
          id="maxAdvance"
          placeholder="advance"
          name="advance"
          value={maxAdvance}
          onChange={e => setMaxAdvance(e.target.value)}
          required
          disabled={isBlocked}
        />
      </div>
    </div>
    <div className="row mb-3">
      <label htmlFor="email" className="col-sm-4 col-form-label">
        Daily Goal
      </label>
      <div className="col-sm-8">
        <input
          className="form-control"
          type="text"
          id="goal"
          placeholder="Daily Goal"
          name="goal"
          value={dgoal}
          onChange={e => setdGoal(e.target.value)}
          required
          disabled={isBlocked}
        />
      </div>
    </div>

    <div className="row mb-3">
      <label htmlFor="email" className="col-sm-4 col-form-label">
        Monthly Goal
      </label>
      <div className="col-sm-8">
        <input
          className="form-control"
          type="text"
          id="goal"
          placeholder="Monthly Goal"
          name="goal"
          value={goal}
          onChange={e => setGoal(e.target.value)}
          required
          disabled={isBlocked}
        />
      </div>
    </div>

    <div className="row mb-3">
      <label htmlFor="email" className="col-sm-4 col-form-label">
        Yearly Goal
      </label>
      <div className="col-sm-8">
        <input
          className="form-control"
          type="text"
          id="goal"
          placeholder="Yearly Goal"
          name="goal"
          value={ygoal}
          onChange={e => setyGoal(e.target.value)}
          required
          disabled={isBlocked}
        />
      </div>
    </div>

    <div className="row mb-3">
      <label htmlFor="email" className="col-sm-4 col-form-label">
        Notices
      </label>
      <div className="col-sm-8">
      <textarea
  className="form-control"
  id="maxAdvance"
  placeholder="advance"
  name="advance"
  value={notice}
  onChange={e => setNotice(e.target.value)}
  required
  disabled={isBlocked}
/>

      </div>
    </div>
  </>



 
 
</div>



     
</div>
</div>
)}
<br></br>

  </div>
 

    
</div>

<div class="row">
<div class="col-sm-4">

</div> 
<div class="col-sm-4">
<center><button onClick={edit} class="success">Edit</button>&nbsp;
<button onClick={update} class="success">Save</button></center>
<br></br><br></br><br></br>
 </div> 
 <div class="col-sm-4">

 </div> 

 </div> 
  </main>





</div>
  

        </>
    )


}
export default Settings;