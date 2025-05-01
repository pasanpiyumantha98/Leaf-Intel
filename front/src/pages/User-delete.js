import '../App.css';
import { useNavigate, useParams } from 'react-router-dom';
import { useState, useEffect } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Header from '../components/Header';

function UserDelete() {
  const [isDeleted, setIsDeleted] = useState(false);
  const [error, setError] = useState(null);
  const { id } = useParams();  // Extract user ID from the URL
  const navigate = useNavigate();  // Hook to programmatically navigate

  useEffect(() => {
    const deleteUser = async () => {
      try {
        const response = await axios.delete(`/api/user-delete/${id}`);
        if (response.status === 200) {
          setIsDeleted(true); // Mark user as deleted
        }
      } catch (err) {
        setError('Error deleting user');
        console.error(err);
      }
    };

    if (id) {
      deleteUser();
    }
  }, [id]);

  useEffect(() => {
    if (isDeleted) {
      // After successful deletion, navigate to the /users page
      setTimeout(() => {
        navigate('/users');
      }, 2000);  // Redirect after 2 seconds
    }
  }, [isDeleted, navigate]);

  return (
    <div className="app-container">
      <Header />
      <main className="content">
        <br /><br />
        <center>
          {error ? (
            <>
              <h1>Error: {error}</h1>
              <a href="/users" className="error nounder">Go Back</a>
            </>
          ) : (
            <>
              {isDeleted ? (
                <>
                  <h1>User Removed successfully!</h1>
                  <a href="/users" className="success nounder">Go Back</a>
                </>
              ) : (
                <h1>Deleting User...</h1>
              )}
            </>
          )}
        </center>
      </main>
    </div>
  );
}

export default UserDelete;
