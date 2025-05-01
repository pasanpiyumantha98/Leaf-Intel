import '../App.css';
import { useState,useEffect } from 'react';
import axios from 'axios';
import React from "react";


function SupplierLotList(props)
{

  const id = props.supplierId;

    const [SuppLot,setSuppLot] = useState([]);

    useEffect(()=>{


        const loadLots = async() =>{
    
            
        const response = await axios.get(`/api/supplier/lots/${id}`);
    
        const lots = response.data;
    
        setSuppLot(lots);
    
        };
    
        loadLots();
    
    },[]);

return (

    <table class="table">
    <thead>
      <tr>
      <th scope="col">Date</th>
        <th scope="col">Quantity</th>
        <th scope="col">Grade</th>
        <th scope="col">Water Level</th>
        <th scope="col">Accepted By</th>
      </tr>
    </thead>
    <tbody>
      {SuppLot.map(LotInfo1 => (
        
           <tr>
           <td>{LotInfo1.Date}/{LotInfo1.Month}</td>
           <td>{LotInfo1.AccQty.toFixed(2)} KG</td>
           <td>{LotInfo1.Grade.toUpperCase()}</td>
           <td>{LotInfo1.Water}</td>
           <td> {LotInfo1.StaffUName}</td>
         </tr>
          
          ))}
   
     
    </tbody>
  </table>

)


}
export default SupplierLotList;