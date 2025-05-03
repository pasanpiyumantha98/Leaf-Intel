import './App.css';
import Header from './components/Header.js';

import Dashboard from './pages/Home.js';
import Login from './pages/Login.js';
import SupReg from './pages/SupReg.js';
import { BrowserRouter,Route,Routes } from 'react-router-dom';
import Settings from './pages/Settings.js';
import Suppliers from './pages/Suppliers.js';
import SignupPage from './pages/Signup.js';
import SupProfile from './pages/SupProfile.js';
import SupDelete from './pages/Supplier-delete.js';
import Accepting from './pages/Accepting.js';
import SampleCheck from './pages/SampleCheck.js';
import RequireAuth from './components/RequireAuth.js';
import Logout from './pages/Logout.js';
import Billing from './pages/Billing.js';
import AdvanceSuccess from './pages/AdvanceSuccess.js';
import Transaction from './pages/Transaction.js';
import PaymentFinal from './pages/PaymentFinal.js';
import Users from './pages/Users.js';
import UserReg from './pages/UserReg.js';
import UserDelete from './pages/User-delete.js';
import UserProfile from './pages/UserProfile.js';
import React from "react";
import SS from './pages/s1.js';
import BarChartExample  from './components/BillingChart.js'
import Login1 from './pages/Login1.js';


function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/login1" element={<Login />} />

        <Route path="/" element={<Login1 />} />

        
    
        
      
    
       

        {/* Protected routes */}
        <Route
          path="/dashboard"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing Clerk", "FO"]}>
              <Dashboard />
            </RequireAuth>
          }
        />
         <Route
          path="/bar"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing Clerk", "FO"]}>
              <BarChartExample />
            </RequireAuth>
          }
        />
        <Route
          path="/suppliers"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing Clerk", "FO"]}>
              <Suppliers />
            </RequireAuth>
          }
        />
        <Route
          path="/settings"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing Clerk", "FO"]}>
              <Settings />
            </RequireAuth>
          }
        />
        <Route
          path="/supplier-registration"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing Clerk", "FO"]}>
              <SupReg />
            </RequireAuth>
          }
        />
        
        <Route
          path="/supplier-profile/:Code"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing Clerk", "FO"]}>
              <SupProfile />
            </RequireAuth>
          }
        />
        <Route
          path="/user-profile/:StaffId"
          element={
            <RequireAuth allowedCategories={["Manager"]}>
              <UserProfile />
            </RequireAuth>
          }
        />
        <Route
          path="/supplier-delete/:id"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing", "FO"]}>
              <SupDelete />
            </RequireAuth>
          }
        />
        <Route
          path="/accepting"
          element={
            <RequireAuth allowedCategories={["Manager", "FO"]}>
              <Accepting/>
            </RequireAuth>
          }
        />
        <Route
          path="/logout"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing", "FO"]}>
              <Logout/>
            </RequireAuth>
          }
        />
        <Route
          path="/accepting/samplecheck/v2/:lotid"
          element={
            <RequireAuth allowedCategories={["Manager", "FO"]}>
              <SampleCheck/>
            </RequireAuth>
          }
        />
        <Route
          path="/billing"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing"]}>
              <Billing/>
            </RequireAuth>
          }
        />
        <Route
          path="/advance-success/:TransId"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing"]}>
              <AdvanceSuccess/>
            </RequireAuth>
          }
        />
         <Route
          path="/transaction/:TransId"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing"]}>
              <Transaction/>
            </RequireAuth>
          }
        />
        <Route
          path="/payment-final"
          element={
            <RequireAuth allowedCategories={["Manager", "Billing"]}>
              <PaymentFinal/>
            </RequireAuth>
          }
        />
        <Route
          path="/users"
          element={
            <RequireAuth allowedCategories={["Manager"]}>
              <Users/>
            </RequireAuth>
          }
        />
        <Route
          path="/users/registration"
          element={
            <RequireAuth allowedCategories={["Manager"]}>
              <UserReg/>
            </RequireAuth>
          }
        />
        <Route
          path="/user-delete/:id"
          element={
            <RequireAuth allowedCategories={["Manager"]}>
              <UserDelete />
            </RequireAuth>
          }
        />
        <Route
          path="/accepting/samplecheck/v1/:lotid"
          element={
            <RequireAuth allowedCategories={["Manager", "FO"]}>
              <SS />
            </RequireAuth>
          }
        />
        
      </Routes>
          


          
    </BrowserRouter>
  );
}

export default App;
