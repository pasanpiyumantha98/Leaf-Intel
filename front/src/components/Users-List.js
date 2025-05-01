import '../App.css';
import { useState,useEffect } from 'react';
import axios from 'axios';
import React from "react";


function UsersList()
{

    const [User,setUsInfo] = useState([]);

    useEffect(()=>{


        const loadSupplier = async() =>{
    
            
        const response = await axios.get(`/api/users`);
    
        const users = response.data;
    
        setUsInfo(users);
    
        };
    
        loadSupplier();
    
    },[]);

return (

    <table class="table">
    <thead>
      <tr>
        <th scope="col">Staff Id</th>
        <th scope="col">Name</th>
        <th scope="col">Access Level</th>
        <th scope="col">Telephone</th>
      </tr>
    </thead>
    <tbody>
      {User.map(User1 => (
           <tr>
           <th scope="row">{User1.StaffId}</th>
           <td>{User1.FirstName}</td>
           <td>{User1.accessLevel}</td>
           <td>{User1.Phone}</td>
          
         </tr>
          
          ))}
   
     
    </tbody>
  </table>

)


}
export default UsersList;