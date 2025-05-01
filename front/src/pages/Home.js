import '../App.css';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';
import { useState,useEffect } from 'react';
import QtyChart from '../components/QtyChart.js';
import Goal from '../components/goal.js';
import { Typography } from '@mui/material';



function Dashboard(){

   
  const [today, setToday] = useState({ totalQtyReceivedToday: 0 });
  const [yesterday, setYesterday] = useState({ totalQtyReceivedyesterday: 0 });
  const [month, setMonth] = useState({ totalQtyReceivedMonth: 0 });
  const [year, setYear] = useState({ totalQtyReceivedYear: 0 });
  const [notice, setNotice] = useState("Loading..");
  const [insight, setInsight] = useState([]);
  const [topSup, setTopSup] = useState([]);
  const [mgoal, setmGoal] = useState("");
  const [dgoal, setdGoal] = useState("");
  const [ygoal, setyGoal] = useState("");
  
  const user = JSON.parse(localStorage.getItem('user')) || {};


  useEffect(()=>{

    const loadTopSup = async()=>{


      const response = await axios.get(`/api/suppliers/top4`);

      const top = response.data;

      setTopSup(top);

      const response3 = await axios.get(`/api/settings/notice`);

      const set2 = response3.data;
  
      setNotice(set2.Value);

      const response4 = await axios.get(`/api/insights`);

      const set3 = response4.data;
  
      setInsight(set3);


    };

    loadTopSup();


  },[]);



  useEffect(()=>{


    const loadtody = async() =>{

        
    const response = await axios.get(`/api/lots/today`);

    const today = response.data;

    
    setToday(today);

    };

    loadtody();

},[]);

useEffect(()=>{


  const loadyesterday = async() =>{

      
  const response = await axios.get(`/api/lots/yesterday`);

  const yesterday = response.data;

  setYesterday(yesterday);

  };

  loadyesterday();

},[]);

useEffect(()=>{


  const loadYear = async() =>{

      
  const response = await axios.get(`/api/lots/year`);

  const year = response.data;

  setYear(year);

  };

  loadYear();

},[]);

useEffect(()=>{


  const loadmonth = async() =>{

      
  const response = await axios.get(`/api/lots/month`);

  const month = response.data;

  setMonth(month);

  const response2 = await axios.get(`/api/settings/mgoal`);

  const goal = response2.data;

  setmGoal(goal.Value);

  const response3 = await axios.get(`/api/settings/dgoal`);

  const dgoal = response3.data;

  setdGoal(dgoal.Value);

  const response4 = await axios.get(`/api/settings/ygoal`);

  const ygoal = response4.data;

  setyGoal(ygoal.Value);

  };

  loadmonth();

},[]);



    return(
        <>

    
   
<div className="app-container">
    <Header/>
  <main className="content">

  <h1 class="pagetitle" >Dashboard</h1>

<div class="row">


  <div class="col-sm-2">

  <div class="card">
    <div class="card-body">
      <h6 class="card-title"><center>Today</center></h6>
      <br/>
      <center><h3 class="figures">{today.totalQtyReceivedToday.toFixed(2)} kg</h3></center>
   </div>
  </div>

  </div>

  <div class="col-sm-2">

   <div class="card">
    <div class="card-body">
      <h6 class="card-title"><center>Yesterday</center></h6>
      <br/>
      <center><h3 class="figures">{yesterday.totalQtyReceivedyesterday.toFixed(2)} kg</h3></center>
   </div>
  </div>

  </div>

  <div class="col-sm-2">

   <div class="card">
    <div class="card-body">
      <h6 class="card-title"><center>This Month</center></h6>
      <br/>
      <center><h3 class="figures">{month.totalQtyReceivedMonth.toFixed(2)} kg</h3></center>
   </div>
  </div>
  
  </div>
  


 
  <div class="col-sm-6">  
    
     <div class="card">
    <div class="card-body">
      <h4 class="card-title"><center>Insights</center></h4>
      <br/>
     <center> <p>{insight}</p> </center>
    </div>
  </div>
  
  </div>
</div>

<div class="row">

<div class="col-sm-6">  
<br></br>
     <div class="card">
    <div class="card-body">
      <h4 class="card-title"><center>Notices</center></h4>
      <p class="card-text">{notice}</p>
    </div>
  </div>
  
  </div>
 

  
  


 
  <div class="col-sm-6">  
  <br/>
  <div class="card">
    <div class="card-body">
      <h4 class="card-title"><center>Leaf Goals</center></h4>
      <div style={{ display: 'flex', justifyContent: 'center', gap: '40px' }}>
  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
    <Goal percentage={Math.min(
      100,                           // Capping the max value to 100              
      Math.max(0, Math.round(                   
        (today.totalQtyReceivedToday / dgoal) * 100
      ))
    )} />
    <p>Daily</p>
  </div>
  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
    <Goal percentage={Math.min(
      100,                                    
      Math.max(0, Math.round(                  
        (month.totalQtyReceivedMonth / mgoal) * 100
      ))
    )} />
    <p>Monthly</p>
  </div>
  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
    <Goal percentage={Math.min(
      100,                                      
      Math.max(0, Math.round(                
        (year.totalQtyReceivedYear / ygoal) * 100
      ))
    )} />
    <p>Yealry</p>
  </div>
</div>


     </div>
  </div>

</div>
</div>

<div class="row">

<div class="col-sm-6">  
<br></br>
<div class="card">
   <div class="card-body">
     <h4 class="card-title"><center>Top Suppliers</center></h4>
     <br/>
     <table class="table">
   <thead>
     <tr>
       <th scope="col">ID</th>
       <th scope="col">Name</th>
       <th scope="col">Member Since</th>
       <th scope="col">Points</th>
       
     </tr>
   </thead>
   <tbody>
     {topSup.map(topSup1 => (
          <tr>
          <th scope="row">{topSup1.Code}</th>
          <td>{topSup1.FirstName}</td>
          <td>{topSup1.RegYear}/{topSup1.RegMonth}</td>
          <td>{topSup1.Points}</td>
          
        </tr>
         
         ))}
  
    
   </tbody>
 </table>
   </div>
 </div>
 
 </div>
  
  
 

  
  


 
  <div class="col-sm-6">  
  <br/>
    <div class="card">
   <div class="card-body">
   <QtyChart/>
   </div>
 </div>
 <br></br> <br></br>
 </div>
</div>



  </main>
</div>

    
       




        </>
    )


}
export default Dashboard;