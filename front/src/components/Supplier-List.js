import '../App.css';
import { useState,useEffect } from 'react';
import axios from 'axios';
import React from "react";


function SupplierList()
{

    const [SupInfo,setSupInfo] = useState([]);

    useEffect(()=>{


        const loadSupplier = async() =>{
    
            
        const response = await axios.get(`/api/suppliers`);
    
        const supplier = response.data;
    
        setSupInfo(supplier);

        
    
        };
    
        loadSupplier();
    
    },[]);

return (

    <table class="table">
    <thead>
      <tr>
        <th scope="col">ID</th>
        <th scope="col">Name</th>
        <th scope="col">NIC</th>
        <th scope="col">Telephone</th>
        <th scope="col">Member Since</th>
        <th scope="col">Ponits</th>
      </tr>
    </thead>
    <tbody>
      {SupInfo.map(SupInfo1 => (
           <tr>
           <th scope="row">{SupInfo1.Code}</th>
           <td>{SupInfo1.FirstName}</td>
           <td>{SupInfo1.NIC}</td>
           <td>{SupInfo1.Phone}</td>
           <td>{SupInfo1.RegYear}</td>
           <td>{SupInfo1.Points}</td>
         </tr>
          
          ))}
   
     
    </tbody>
  </table>

)


}
export default SupplierList;