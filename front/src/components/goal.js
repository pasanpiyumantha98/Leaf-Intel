import React from 'react';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

const Goal = ({ percentage }) => {
  return (
    <div style={{ width: 120, height: 120 }}>
      <CircularProgressbar
        value={percentage}
        text={`${percentage}%`}
        styles={buildStyles({
          textColor: '#333',
          pathColor: '#4caf50',
          trailColor: '#d6d6d6',
        })}
      />
    </div>
  );
};

export default Goal;
