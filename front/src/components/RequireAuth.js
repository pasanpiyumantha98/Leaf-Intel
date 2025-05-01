// src/components/RequireAuth.jsx
import React from 'react';
import { Navigate } from 'react-router-dom';

function RequireAuth({ children, allowedCategories }) {
  // Here, replace localStorage with your actual auth logic:
  const isAuthenticated = localStorage.getItem('token');

  const user = JSON.parse(localStorage.getItem('user'));

  // If not authenticated, redirect to the login page ("/").
  if (!isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  if (!allowedCategories.includes(user.accessLevel)) {
    return <Navigate to="/home" replace />;
  }

  // If authenticated, allow access to the protected page/component
  return children;
}

export default RequireAuth;
