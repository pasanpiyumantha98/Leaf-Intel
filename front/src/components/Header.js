import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faBars,
  faHome,
  faUserCheck,
  faToolbox,
  faPeopleGroup,
  faHandHolding,
  faMoneyBill,
  faSignOut
} from '@fortawesome/free-solid-svg-icons';
import '../vnavbar.css';
import logo from '../img/logo1.png';

function Header() {
  const user = JSON.parse(localStorage.getItem('user')) || {};

  // State to handle sidebar open/close on mobile
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="app-container">
      {/* 1) MOBILE TOP BAR (hamburger + brand) */}
      <header className="mobile-header">
        <div className="mobile-logo">
          <img src={logo} alt="Logo" />
        </div>
        <button className="hamburger-btn" onClick={toggleSidebar}>
          <FontAwesomeIcon icon={faBars} />
        </button>
      </header>

      {/* 2) SIDEBAR (collapsed by default on mobile) */}
      <nav className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        {/* Desktop Logo (optional): Hide on mobile if you like */}
        <div className="logo">
          <img src={logo} alt="Logo" />
        </div>

        <div className="nav-links">
          <a href="/dashboard" className="nav-item">
            <FontAwesomeIcon icon={faHome} className="icon" />
            <span>Dashboard</span>
          </a>

          {(user.accessLevel === "Manager" || user.accessLevel === "FO") && (
            <a href="/accepting" className="nav-item">
              <FontAwesomeIcon icon={faHandHolding} className="icon" />
              <span>Accepting</span>
            </a>
          )}

          <a href="/suppliers" className="nav-item">
            <FontAwesomeIcon icon={faPeopleGroup} className="icon" />
            <span>Suppliers</span>
          </a>

          {(user.accessLevel === "Manager" || user.accessLevel === "Billing") && (
            <a href="/billing" className="nav-item">
              <FontAwesomeIcon icon={faMoneyBill} className="icon" />
              <span>Billing</span>
            </a>
          )}

          {user.accessLevel === "Manager" && (
            <a href="/users" className="nav-item">
              <FontAwesomeIcon icon={faUserCheck} className="icon" />
              <span>Users</span>
            </a>
          )}

          
            <a href="/Settings" className="nav-item">
              <FontAwesomeIcon icon={faToolbox} className="icon" />
              <span>Settings</span>
            </a>
          

          <a href="/logout" className="nav-item">
            <FontAwesomeIcon icon={faSignOut} className="icon" />
            <span>Log Out</span>
          </a>
        </div>
      </nav>

      
    </div>
  );
}

export default Header;
