
import { Link,useNavigate } from 'react-router-dom';
import React, { useEffect } from 'react';

function Logout()
{

const navigate = useNavigate();

useEffect(() => {
    // Remove any stored authentication data
    localStorage.removeItem('token');
    
    // Automatically redirect to the login page (or wherever you'd like)
    navigate('/');
  }, [navigate]);
  

}
export default Logout;