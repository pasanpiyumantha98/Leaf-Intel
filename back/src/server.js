import express from "express";
import { MongoClient,ServerApiVersion } from "mongodb";
const uri = "mongodb+srv://pasanpiyumantha98:G30JMklKNHdfum19@cluster0.etw2b.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0";
import multer from "multer";
import bcrypt from 'bcrypt';
       

import twilio from 'twilio';
const clientt  = twilio(
  'AC2bb1e85e5c0fa7ed6c26890294a74f61',
 '271f423fe1218461ae83cfd03b805296'
);


const upload = multer({ storage: multer.memoryStorage() });
const SALT_ROUNDS = 10;  

/// START Database SetUp
const client = new MongoClient(uri, {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
  },
});

await client.connect();
const db = client.db('LeafIntel');
/// END Database SetUp


const app= express();
app.use(express.json());

app.get('/hello',(req,res)=>{

res.send("Hello");


});

app.post('/whatsapp-send/go', async (req, res) => {
  try {
    const { to, body } = req.body;               // ① grab data from front-end
    if (!to || !body) return res.status(400).send('Missing to/body');

    const msg = await clientt.messages.create({   // ② call Twilio
      from: 'whatsapp:+14155238886',             // sandbox number
      to:   `whatsapp:${to}`,                    // user’s phone
      body                                       // text
    });

    return res.json({ sid: msg.sid });           // ③ send SID back
  } catch (err) {
    console.error(err);
    return res.status(500).send('WHATSAPP_SEND_FAILED');
  }
});


/// Create Supplier
app.post('/suppliers-registration/go',upload.single('pic'), async(req,res)=>{
  try {
    
    const { fname, lname, email, city, phone, nic,file } = req.body;

 
    const currentDate = new Date();
    const date = currentDate.getDate();
    const month = currentDate.getMonth() + 1; 
    const year = currentDate.getFullYear();

    const fileBuffer = req.file ? req.file.buffer : null;

 
    const existingSupplier = await db.collection('Suppliers').findOne({ NIC: nic });

    if (existingSupplier) {
 
      return res.send('NOTOK');
    }

    const LastSupplier = await db.collection('Suppliers')
  .findOne({}, { sort: { Code: -1 } });

   const code = LastSupplier.Code + 2;
   
    await db.collection('Suppliers').insertOne({
      FirstName: fname,
      LastName: lname,
      Pic:fileBuffer,
      Email: email,
      City: city,
      Phone: phone,
      NIC: nic,
      RegDate: date,
      RegMonth: month,
      RegYear: year,
      Code: code,
      Points:0
    });

  
    return res.send('OK');
    
  } catch (error) {
    console.error(error);
    return res.status(500).send('Internal Server Error');
  }
});

app.get('/api/supplier/:id/pic', async (req, res) => {
  try {
    // 1) Grab the param
    const supplierCode = parseInt(req.params.id, 10);

    // 2) Find the supplier
    const supplier = await db.collection('Suppliers').findOne({ Code: supplierCode });
    if (!supplier) {
      return res.status(404).send('Supplier not found');
    }
    if (!supplier.Pic) {
      return res.status(404).send('No image found for this supplier');
    }

    // 3) Check if it's a BSON Binary object
    // If your console.log(supplier.Pic) reveals something like:
    // { _bsontype: 'Binary', sub_type: 0, position: 1234, buffer: <Buffer ...> }
    // you need to extract .buffer
    let picBuffer;

    // If using the native MongoDB driver or Mongoose, you might do:
    if (supplier.Pic.buffer) {
      picBuffer = supplier.Pic.buffer; 
    } else {
      // If it is already a Buffer, for example, if you stored it with Mongoose
      picBuffer = supplier.Pic; 
    }

    // 4) Optionally determine correct mime type if stored, e.g.:
    // res.set('Content-Type', supplier.PicMimeType || 'image/jpeg');

    res.set('Content-Type', 'image/jpeg');
    return res.send(picBuffer);

  } catch (error) {
    console.error(error);
    return res.status(500).send('Server error');
  }
});



/// Create User
app.post('/users-registration/go',async(req,res)=>{
  try {
    
    const { fname, lname, email, city, phone, nic,accesslevel,gender,uName,pass } = req.body;

 
    const currentDate = new Date();
    const date = currentDate.getDate();


 
    const existingUser = await db.collection('Users').findOne({ NIC: nic });

    if (existingUser) {
 
      return res.send('UserExist');
    }

    const UserName = await db.collection('Users').findOne({ UserName: uName });

    if (UserName) {
 
      return res.send('UserNameExist');
    }

    const LastUser = await db.collection('Users')
  .findOne({}, { sort: { Code: -1 } });

   const code = parseInt(LastUser.StaffId) + 5;


   const hashedPass = await bcrypt.hash(pass, SALT_ROUNDS);

   
    await db.collection('Users').insertOne({
      FirstName: fname,
      LastName: lname,
      Email: email,
      City: city,
      Phone: phone,
      NIC: nic,
      Date: date,
      StaffId: code,
      accessLevel: accesslevel,
      Gender : gender,
      UserName: uName,
      Password: hashedPass,
    });

  
    return res.send('OK');
    
  } catch (error) {
    console.error(error);
    return res.status(500).send('Internal Server Error');
  }
});


/// Search Supplier
app.get('/api/supplier/:id', async(req,res) =>{

  try {

    const id = parseInt(req.params.id);

    const supplier = await db.collection('Suppliers').findOne({ Code: id });

    if (!supplier) {
      
      return res.status(404).json({ message: 'Supplier not found' });
    }

    return res.json(supplier);
    
  } catch (err) {
 
    console.error('Error fetching supplier:', err);
    return res.status(500).json({ message: 'Internal Server Error' });
  }
  
  });

  /// Search User
app.get('/api/user/:id', async(req,res) =>{

  try {

    const id = parseInt(req.params.id);

    const user = await db.collection('Users').findOne({ StaffId: id });

    if (!user) {
      
      return res.status(404).json({ message: 'Supplier not found' });
    }

    return res.json(user);
    
  } catch (err) {
 
    console.error('Error fetching supplier:', err);
    return res.status(500).json({ message: 'Internal Server Error' });
  }
  
  });



/// Supplier Delete Function
  app.get('/api/supplier-delete/:id', async(req,res) =>{
    try {
      const id = req.params.id;
      const result = await db.collection('Suppliers').deleteOne({ NIC: id });
      res.json(result);
    } catch (err) {
      console.error(err);
      res.status(500).send('Error deleting supplier');
    }
  
  });

  /// User Delete Function
  app.get('/api/user-delete/:id', async(req,res) =>{
    try {
      const id = parseInt(req.params.id);
      const result = await db.collection('Users').deleteOne({ StaffId: id });
      res.json(result);
    } catch (err) {
      console.error(err);
      res.status(500).send('Error deleting supplier');
    }
  
  });

  // User Delete Function
app.delete('/api/user-delete/:id', async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    const result = await db.collection('Users').deleteOne({ StaffId: id });
    
    if (result.deletedCount === 0) {
      return res.status(404).send('User not found');
    }
    
    res.json({ message: 'User deleted successfully', result });
  } catch (err) {
    console.error(err);
    res.status(500).send('Error deleting user');
  }
});



/// Supplier List Function
  app.get('/api/suppliers', async(req,res) =>{
    try {
      const suppliers = await db.collection('Suppliers').find({}).toArray();
      res.json(suppliers);
    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });

  /// Settings Function Rate
  app.get('/api/settings/rate', async(req,res) =>{
    try {
      const set = await db.collection('AppSettings').findOne({ Name: "Rate" });
      res.json(set);
    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });

  /// Settings Function Goal
  app.get('/api/settings/mgoal', async(req,res) =>{
    try {
      const set = await db.collection('AppSettings').findOne({ Name: "MonthlyGoal" });
      res.json(set);
    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });

   /// Settings Function Goal Yearly
   app.get('/api/settings/ygoal', async(req,res) =>{
    try {
      const set = await db.collection('AppSettings').findOne({ Name: "YearlyGoal" });
      res.json(set);
    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });

   /// Settings Function Goal Daily
   app.get('/api/settings/dgoal', async(req,res) =>{
    try {
      const set = await db.collection('AppSettings').findOne({ Name: "DailyGoal" });
      res.json(set);
    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });

    /// Settings Function Max Advance
    app.get('/api/settings/maxadvance', async(req,res) =>{
      try {
        const set = await db.collection('AppSettings').findOne({ Name: "MaxAdvance" });
        res.json(set);
      } catch (error) {
        console.error(error);
        res.status(500).send('Internal server error');
      }
    });

    /// Settings Function Notice
    app.get('/api/settings/notice', async(req,res) =>{
      try {
        const set = await db.collection('AppSettings').findOne({ Name: "Notice" });
        res.json(set);
      } catch (error) {
        console.error(error);
        res.status(500).send('Internal server error');
      }
    });


  /// User List Function
  app.get('/api/users', async(req,res) =>{
    try {
      const users = await db.collection('Users').find({}).toArray();
      res.json(users);
    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });


  /// Lot List Function
  app.get('/api/lots', async(req,res) =>{
    try {
     
      const lots = await db.collection('LeafLots')
  .find({Stat: 1, Stat:2})
  .sort({ _id: -1 }) // Sort descending by _id
  .limit(100)         // Get only the latest 10 documents
  .toArray();


      res.json(lots);
    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });

   /// Supplier Lot List Function
   app.get('/api/supplier/lots/:id', async(req,res) =>{
    try {

      const id = parseInt(req.params.id);  

      const today = new Date();
      const currentMonth = today.getMonth() + 1; 
     
      const lots = await db.collection('LeafLots')
  .find({
    SuppId: id,
    Month: currentMonth,
    Stat: 1,
    Stat: 2 
  })
  .sort({ _id: -1 }) 
  .toArray();


      res.json(lots);
    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });

   /// Today QTY Function
   app.get('/api/lots/today', async(req,res) =>{
    try {

const today = new Date();
const currentYear = today.getFullYear();
const currentMonth = today.getMonth() + 1; 
const currentDay = today.getDate();

      
//pipeline.
const pipeline = [
  {
    
    $match: {
      Year: currentYear,
      Month: currentMonth,
      Date: currentDay, 
      Stat: { $in: [1, 2] }
      
    },
  },
  {
    
    $group: {
      _id: null,
      totalAccQty: { $sum: '$AccQty' } 
    },
  },
];


const results = await db.collection('LeafLots').aggregate(pipeline).toArray();
const totalQtyReceivedToday = results[0]?.totalAccQty || 0;

res.json({ totalQtyReceivedToday });



    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });

   /// Year QTY Function
   app.get('/api/lots/year', async(req,res) =>{
    try {

const today = new Date();
const currentYear = today.getFullYear();
const currentMonth = today.getMonth() + 1; 
const currentDay = today.getDate();

      
//pipeline.
const pipeline = [
  {
    
    $match: {
      Year: currentYear,
      Stat: { $in: [1, 2] }
      
      
    },
  },
  {
    
    $group: {
      _id: null,
      totalAccQty: { $sum: '$AccQty' } 
    },
  },
];


const results = await db.collection('LeafLots').aggregate(pipeline).toArray();
const totalQtyReceivedYear = results[0]?.totalAccQty || 0;

res.json({ totalQtyReceivedYear });



    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });

  /// Yesterday QTY Function
    app.get('/api/lots/yesterday', async(req,res) =>{
      try {
  
  const today = new Date();
  let currentYear = today.getFullYear();
  let currentMonth = today.getMonth() + 1; 
  let Yesterday = today.getDate()-1;

 

  if(Yesterday===0)
  {
    currentMonth = currentMonth -1;

    if(currentMonth===0)
    {
      currentMonth=12
      currentYear=currentYear-1
    }

    if (currentMonth === 1)
      {
        Yesterday += 31;
      } 
      else if (currentMonth === 2)
      {
        let bal = currentYear % 4;
      
        if (bal === 0)
        {
          Yesterday += 29;
        } 
        else
        {
          Yesterday += 28;
        }
      } 
      else if (currentMonth === 3)
      {
        Yesterday += 31;
      }
      else if (currentMonth === 4)
      {
        Yesterday += 30;
      }
      else if (currentMonth === 5)
      {
        Yesterday += 31;
      }
      else if (currentMonth === 6)
      {
        Yesterday += 30;
      }
      else if (currentMonth === 7)
      {
        Yesterday += 31;
      }
      else if (currentMonth === 8)
      {
        Yesterday += 31;
      }
      else if (currentMonth === 9)
      {
        Yesterday += 30;
      }
      else if (currentMonth === 10)
      {
        Yesterday += 31;
      }
      else if (currentMonth === 11)
      {
        Yesterday += 30;
      }
      else 
      {
        Yesterday += 31;
      }
      


  }
  
        
  //pipeline.
  const pipeline = [
    {
      
      $match: {
        Year: currentYear,
        Month: currentMonth,
        Date: Yesterday, 
        Stat: { $in: [1, 2] }
         
      },
    },
    {
      
      $group: {
        _id: null,
        totalAccQty: { $sum: '$AccQty' } 
      },
    },
  ];
  
  
  const results = await db.collection('LeafLots').aggregate(pipeline).toArray();
  const totalQtyReceivedyesterday = results[0]?.totalAccQty || 0;
  
  res.json({ totalQtyReceivedyesterday });
  
  
  
      } catch (error) {
        console.error(error);
        res.status(500).send('Internal server error');
      }
    }); 


/// Month QTY Function
   app.get('/api/lots/month', async(req,res) =>{
    try {

const today = new Date();
const currentYear = today.getFullYear();
const currentMonth = today.getMonth() + 1; 


      
//pipeline.
const pipeline = [
  {
    
    $match: {
      Year: currentYear,
      Month: currentMonth,
      
    },
  },
  {
    
    $group: {
      _id: null,
      totalAccQty: { $sum: '$AccQty' } 
    },
  },
];


const results = await db.collection('LeafLots').aggregate(pipeline).toArray();
const totalQtyReceivedMonth = results[0]?.totalAccQty || 0;

res.json({ totalQtyReceivedMonth });



    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });  


  /// Supplier Monthly QTY Function
  app.get('/api/supplier/qty/:id', async(req,res) =>{
    try {

const id = parseInt(req.params.id);      
const today = new Date();
const currentYear = today.getFullYear();
const currentMonth = today.getMonth() + 1; 


      
//pipeline.
const pipeline = [
  {
    
    $match: {
      Year: currentYear,
      Month: currentMonth,
      SuppId: id
    },
  },
  {
    
    $group: {
      _id: null,
      totalAccQty: { $sum: '$AccQty' } 
    },
  },
];


const results = await db.collection('LeafLots').aggregate(pipeline).toArray();
const totalQtySupplierMonth = results[0]?.totalAccQty || 0;

res.json({ totalQtySupplierMonth });



    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });  


  /// Supplier All Time QTY Function
  app.get('/api/supplier/qty/all/:id', async(req,res) =>{
    try {

const id = parseInt(req.params.id);      



      
//pipeline.
const pipeline = [
  {
    
    $match: {
      
      SuppId: id
    },
  },
  {
    
    $group: {
      _id: null,
      totalAccQty: { $sum: '$AccQty' } 
    },
  },
];


const results = await db.collection('LeafLots').aggregate(pipeline).toArray();
const totalQtySupplierAll = results[0]?.totalAccQty || 0;

res.json({ totalQtySupplierAll });



    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });  


/// Check Advance Eligibility and recording to payments
app.get('/api/advance/elg/:id/:RqstValue/:un', async (req, res) => {
  try {
    const id = parseInt(req.params.id);  
    const RqstValue = parseFloat(req.params.RqstValue);  
    const staff = req.params.un;  
    const setname = "MaxAdvance";
    const type="Advance";
      

 
    const currentDate = new Date();
    const date = currentDate.getDate();
    const month = currentDate.getMonth() + 1; 
    const year = currentDate.getFullYear();

    const person = await db.collection('Suppliers').findOne({ Code: id});

    if (person) {
      
      const limit = await db.collection('AppSettings').findOne({ Name: setname});

      if(limit.Value>=RqstValue)
      {  

        const pipeline = [
          {
            
            $match: {
              Year: year,
              Month: month,
              SuppId: id, 
              Type:"Advance"
            },
          },
          {
            
            $group: {
              _id: null,
              Amount: { $sum: '$Amount' } 
            },
          },
        ];
        
        
        const results = await db.collection('Payments').aggregate(pipeline).toArray();
        const totAmount = results[0]?.Amount || 0;          

        const AvalbeValue = limit.Value-totAmount;       





        if(RqstValue<=AvalbeValue)
        {
        const Payments = await db.collection('Payments')
        .findOne({}, { sort: { TransId: -1 } });
      
         const TransId = Payments.TransId + 9;
         
          await db.collection('Payments').insertOne({
            TransId: TransId,
            SuppId: id,
            Type: type,
            Amount: RqstValue,
            Date: date,
            Month: month,
            Year: year,
            Staff: staff
          });
          res.send(String(TransId));
        } else {
          res.send("CantGive-notavlble");

        }
       
      } else
      {
        res.send("Higher");
      }



    } else {
      res.send("NoSupplier");
    }
  } catch (error) {
    console.error(error);
    res.status(500).send("Internal Server Error");
  }
});  



/// User login
app.post('/login/go', async (req, res) => {
  try {
    const { uname, pass } = req.body;
// Searching the database for possible matches
    const user = await db.collection('Users').findOne({ UserName: uname });
    if (!user) return res.send('NotOK');         
// Comparing the user input along with encrypted password
    const match = await bcrypt.compare(pass, user.Password);
    if (!match) return res.send('NotOK');       
  
    return res.json(user);          

  } catch (err) {
    console.error(err);
    res.status(500).send('Internal Server Error');
  }
});
  

  /// Accepting 1st stage Function
  app.post('/accepting/first', async (req, res) => {
    try {
    
const SuppId = parseInt(req.body.SuppID, 10);
const Qty = parseFloat(req.body.Qty);
const Range = parseInt(req.body.Range, 10);

const Grade = "NR";


let rangevalue;

if (Range === 1) {
  rangevalue = 0.9;
} else if (Range === 2) {
  rangevalue = 0.85;
} else if (Range === 3) {
  rangevalue = 0.8;
} else if (Range === 4) {
  rangevalue = 0.75;
} else if (Range === 5) {
  rangevalue = 0.7;
} else {
 
  rangevalue = 1; 
}


const AccQty = Qty * rangevalue;




      const StaffUName="Pasan12";

    const currentDate = new Date();
    const Datee = currentDate.getDate();
    const Month = currentDate.getMonth() + 1; 
    const Year = currentDate.getFullYear();

      const Supplier = await db.collection('Suppliers').findOne({ Code: SuppId});
  
      if (Supplier) {
       
        const LastLot = await db.collection('LeafLots')
        .findOne({}, { sort: { LotId: -1 } });
      
         const LotId = LastLot.LotId + 3;

        await db.collection('LeafLots').insertOne({
          SuppId: SuppId,
          Qty: Qty,
          Water: Range,
          AccQty: AccQty,
          Grade: Grade,
          Year: Year,
          Month: Month,
          Date: Datee,
          SuppName: Supplier.FirstName,
          StaffUName: StaffUName,
          LotId: LotId,
          Stat: 0
        });

        
        return res.send({
          message:"OK",
          lotid:LotId
      });

      } else {
        res.send("SuppNotFound");
      }
    } catch (error) {
      console.error(error);
      res.status(500).send("Internal Server Error");
    }
  });


  /// Recording Advances to payments
app.post('/api/supplier/issueadvance',async(req,res)=>{
  try {
    
    const { SuppId, amount } = req.body;
    const type="Advance";
    const staff="Pasan12";

 
    const currentDate = new Date();
    const date = currentDate.getDate();
    const month = currentDate.getMonth() + 1; 
    const year = currentDate.getFullYear();


    const Payment = await db.collection('Payments')
  .findOne({}, { sort: { Code: -1 } });

   const TransId = Payment.TransId + 9;
   
    await db.collection('Payments').insertOne({
      TransId: TransId,
      SuppId: SuppId,
      Type: type,
      Amount: amount,
      Date: date,
      Month: month,
      Year: year,
      Staff: staff
    });

  
    return res.send('OK');
    
  } catch (error) {
    console.error(error);
    return res.status(500).send('Internal Server Error');
  }
});

/// Search Transactions
app.get('/api/transaction/:id', async(req,res) =>{

  try {

    const id = parseInt(req.params.id);

    const trans = await db.collection('Payments').findOne({ TransId: id });

    if (!trans) {
      
      return res.send("TransactionNotFound");
    }

    return res.json(trans);
    
  } catch (err) {
 
    console.error('Error fetching supplier:', err);
    return res.status(500).json({ message: 'Internal Server Error' });
  }
  
  });


    /// Calculating Payable
    app.get('/api/supplier/payment/:id', async (req, res) => {
      try {
        const currentDate = new Date();
        let m = currentDate.getMonth(); 
        let y = currentDate.getFullYear();

        let mName="start"

        if (m === 0) {
          mName = "December";
          y = y - 1;
          m=12;
        } else if (m === 1) {
          mName = "January";
        } else if (m === 2) {
          mName = "February";
        } else if (m === 3) {
          mName = "March";
        } else if (m === 4) {
          mName = "April";
        } else if (m === 5) {
          mName = "May";
        } else if (m === 6) {
          mName = "June";
        } else if (m === 7) {
          mName = "July";
        } else if (m === 8) {
          mName = "August";
        } else if (m === 9) {
          mName = "September";
        } else if (m === 10) {
          mName = "October";
        } else if (m === 11) {
          mName = "November";
        }
        


    
        const SuppId = parseInt(req.params.id);
          
        const Supplier = await db.collection('Suppliers').findOne({ Code: SuppId });
    
        if (!Supplier) {
          return res.send("SupplierNotFound");
        } else {
          // Calculating Monthly QTY
          const pipeline = [
            {
              $match: {
                Year: y,
                Month: m,
                SuppId: SuppId
              }
            },
            {
              $group: {
                _id: null,
                totalAccQty: { $sum: '$AccQty' }
              }
            }
          ];
          
          const results = await db.collection('LeafLots').aggregate(pipeline).toArray();
          const Qty = results[0]?.totalAccQty || 0;
    
          // Extracting rate
          const set = await db.collection('AppSettings').findOne({ Name: "Rate" });
          let Payable = Qty * set.Value;


          const Payment = await db.collection('Payments').findOne({ SuppId: SuppId, PayYear:y, PayMonth:mName, Type:"Monthly Payment" });

          const Advance = await db.collection('Payments').findOne({ SuppId: SuppId, Year:y, Month:m, Type:"Advance" });

          if(Payment)
          {
            return res.send("AlreadyIssued");
          } else if (Advance)
          {
            Payable = Payable - Advance.Amount
            return res.json(Payable);
          } else
          {
            return res.json(Payable);
          }
    
          
        }
      } catch (error) {
        console.error(error);
        return res.status(500).send('Internal Server Error');
      }
    });
    
  

  /// Monthly Pay transaction entry
app.post('/api/supplier/makepay',async(req,res)=>{
  try {
    
  const { SuppId,Payable,Month,Year,un } = req.body;

  const type="Monthly Payment";
 
  const currentDate = new Date();
  const date = currentDate.getDate();
  const month = currentDate.getMonth() + 1; 
  const year = currentDate.getFullYear();


  const Payment = await db.collection('Payments')
  .findOne({}, { sort: { TransId: -1 } });

   const TransId = Payment.TransId + 9;
   
   await db.collection('Payments').insertOne({
    TransId: TransId,
    SuppId: parseInt(SuppId),
    Type: type,
    Amount: Payable,
    Date: date,
    Month: month,
    Year: year,
    PayMonth:Month,
    PayYear:Year,
    Staff: un
  });

  
    return res.json(TransId);
    
  } catch (error) {
    console.error(error);
    return res.status(500).send('Internal Server Error');
  }
});


  /// Update settings
  app.post('/settings/update',async(req,res)=>{
    try {
      
    const { uname,email,password,address,rate,maxadvance,staffid,notice,goal,dgoal,ygoal } = req.body;
    
    const currentDate = new Date();
    
    const filter = { StaffId: staffid };
   
    let updateFields = {
      UserName: uname,
      Email: email,
      City: address,
    };

    if (password.trim() !== "") {
      const hashedPass = await bcrypt.hash(password, SALT_ROUNDS);
      updateFields.Password = hashedPass;
    }

    const result = await db.collection("Users").updateOne(
      filter,
      { $set: updateFields },
    );

    const result1 = await db.collection("AppSettings").updateOne(
      { Name: "MaxAdvance" }, // Filter
      { $set: { Value: maxadvance }  }                // Update
    );

    const result2 = await db.collection("AppSettings").updateOne(
      { Name: "Rate" }, // Filter
      { $set: { Value: rate }  }                // Update
    );

    const result3 = await db.collection("AppSettings").updateOne(
      { Name: "Notice" }, // Filter
      { $set: { Value: notice }  }                // Update
    );
    const result4 = await db.collection("AppSettings").updateOne(
      { Name: "MonthlyGoal" }, // Filter
      { $set: { Value: goal }  }                // Update
    );
    const result5 = await db.collection("AppSettings").updateOne(
      { Name: "YearlyGoal" }, // Filter
      { $set: { Value: ygoal }  }                // Update
    );
    const result6 = await db.collection("AppSettings").updateOne(
      { Name: "DailyGoal" }, // Filter
      { $set: { Value: dgoal }  }                // Update
    );
    
    
      return res.send("ok");
      
    } catch (error) {
      console.error(error);
      return res.status(500).send('Internal Server Error');
    }
  });

  // update accordingly sample check

    app.post('/samplecheck/update',async(req,res)=>{
      try {
        
      const { grade,lotid, points } = req.body;
       
      const filter = { LotId: lotid };
     
      const updateFields = {
        Grade: grade,
        Stat:1
      };
  
      const result = await db.collection("LeafLots").updateOne(
        filter,
        { $set: updateFields },
      );
      

      const Lot = await db.collection('LeafLots').findOne({ LotId: lotid });

      const Supplier = await db.collection('Suppliers').findOne({ Code: Lot.SuppId });

      const newPoints = Supplier.Points+points;

      const filter2 = { Code : Lot.SuppId };

      const updateFields2 = {
        Points: newPoints,
      };
  
      const result2 = await db.collection("Suppliers").updateOne(
        filter2,
        { $set: updateFields2 },
      );
      
        return res.send("ok");
        
      } catch (error) {
        console.error(error);
        return res.status(500).send('Internal Server Error');
      }
    });
    

    /// Update settings
  app.post('/settings/update',async(req,res)=>{
    try {
      
    const { uname,email,password,address,rate,maxadvance,staffid,notice,goal,dgoal,ygoal } = req.body;
    
    const currentDate = new Date();
    
    const filter = { StaffId: staffid };
   
    let updateFields = {
      UserName: uname,
      Email: email,
      City: address,
    };

    if (password.trim() !== "") {
      const hashedPass = await bcrypt.hash(password, SALT_ROUNDS);
      updateFields.Password = hashedPass;
    }

    const result = await db.collection("Users").updateOne(
      filter,
      { $set: updateFields },
    );

    const result1 = await db.collection("AppSettings").updateOne(
      { Name: "MaxAdvance" }, // Filter
      { $set: { Value: maxadvance }  }                // Update
    );

    const result2 = await db.collection("AppSettings").updateOne(
      { Name: "Rate" }, // Filter
      { $set: { Value: rate }  }                // Update
    );

    const result3 = await db.collection("AppSettings").updateOne(
      { Name: "Notice" }, // Filter
      { $set: { Value: notice }  }                // Update
    );
    const result4 = await db.collection("AppSettings").updateOne(
      { Name: "MonthlyGoal" }, // Filter
      { $set: { Value: goal }  }                // Update
    );
    const result5 = await db.collection("AppSettings").updateOne(
      { Name: "YearlyGoal" }, // Filter
      { $set: { Value: ygoal }  }                // Update
    );
    const result6 = await db.collection("AppSettings").updateOne(
      { Name: "DailyGoal" }, // Filter
      { $set: { Value: dgoal }  }                // Update
    );
    
    
      return res.send("ok");
      
    } catch (error) {
      console.error(error);
      return res.status(500).send('Internal Server Error');
    }
  });

  // update transfer stat on lots

    app.get('/api/transfer',async(req,res)=>{
      try {
       
            
      const filter = { Stat: 1 };
     
      const updateFields = {
        Stat: 2
           };
  
      const result = await db.collection("LeafLots").updateMany(
        filter,
        { $set: updateFields },
      );
    
       return res.send("ok");
        
      } catch (error) {
        console.error(error);
        return res.status(500).send('Internal Server Error');
      }
    });
    

    /// Search lots
app.get('/api/lotsfind/:id', async(req,res) =>{

  try {

    const id = parseInt(req.params.id);

    const lot = await db.collection('LeafLots').findOne({ LotId: id });

    if (!lot) {
      
      return res.send("LotNotFound");
    }

    return res.json(lot);
    
  } catch (err) {
 
    console.error('Error fetching supplier:', err);
    return res.status(500).json({ message: 'Internal Server Error' });
  }
  
  });


  /// Top Supplier List Function
  app.get('/api/suppliers/top4', async(req,res) =>{
    try {
      const suppliers = await db.collection('Suppliers').find({}).sort({Points:-1}).limit(4).toArray();
      res.json(suppliers);

    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });

    /// Billing chart final payments
    app.get('/api/billing/chart', async (req, res) =>{
      try {
        const currentDate = new Date();
        const date = currentDate.getDate();
        let month = currentDate.getMonth(); 
        let year = currentDate.getFullYear();
    
        const tots = [];
    
        const start = month - 2;
        const finish = month + 1;
    
        for (let id = start; id <= finish; id++) {
          let mid = id;
          let miy = year;
    
          // If id <= 0 => shift to previous year
          if (id <= 0) {
            mid += 12;
            miy -= 1;
          }
    
          const pipeline = [
            {
              $match: {
                Year: miy,
                Month: mid
              }
            },
            {
              $group: {
                _id: null,
                totalAccQty: { $sum: "$Amount" }
              }
            }
          ];
    
          const results = await db.collection("Payments").aggregate(pipeline).toArray();
          
          // Safely handle totalAccQty
          const Qty = results[0]?.totalAccQty || 0; 
          
    
          let monthName = "";
          if (mid === 1) monthName = "January";
          else if (mid === 2) monthName = "February";
          else if (mid === 3) monthName = "March";
          else if (mid === 4) monthName = "April";
          else if (mid === 5) monthName = "May";
          else if (mid === 6) monthName = "June";
          else if (mid === 7) monthName = "July";
          else if (mid === 8) monthName = "August";
          else if (mid === 9) monthName = "September";
          else if (mid === 10) monthName = "October";
          else if (mid === 11) monthName = "November";
          else if (mid === 12) monthName = "December";
          else monthName = "Invalid month";
    
          tots.push({
            month: monthName,
            Amount: Qty,
          });
        }
    
        res.json(tots);
    
      } catch (error) {
        console.error(error);
        res.status(500).send('Internal server error');
      }
    });


    /// Billing chart final payments
    app.get('/api/qty/chart', async (req, res) =>{
      try {
        const currentDate = new Date();
        let date = currentDate.getDate();;
        let month = currentDate.getMonth(); 
        let year = currentDate.getFullYear();
    
        const tots = [];
    
        const start = date - 5;
        const finish = date;
    
        for (let id = start; id < finish; id++) {
          let mid = id;
          let mimonth = month;

          if (mid<0)
          {
            const isLeapYear = (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
            let days;
  
            if (month === 1) {
              days = 31; 
            } else if (month === 2) {
              days = isLeapYear ? 29 : 28;
            } else if (month === 3) {
              days = 31; // March
            } else if (month === 4) {
              days = 30; // April
            } else if (month === 5) {
              days = 31; // May
            } else if (month === 6) {
              days = 30; // June
            } else if (month === 7) {
              days = 31; // July
            } else if (month === 8) {
              days = 31; // August
            } else if (month === 9) {
              days = 30; // September
            } else if (month === 10) {
              days = 31; // October
            } else if (month === 11) {
              days = 30; // November
            } else  {
              days = 31; // December
            }


            mid = mid + days +1;
            mimonth = mimonth -1;
          }
    
         
    
          const pipeline = [
            {
              $match: {
                Year: year,
                Month: mimonth + 1,
                Date: mid
              }
            },
            {
              $group: {
                _id: null,
                totalAccQty: { $sum: "$AccQty" }
              }
            }
          ];
    
          const results = await db.collection("LeafLots").aggregate(pipeline).toArray();
          
          // Safely handle totalAccQty
          const Qty = results[0]?.totalAccQty || 0; 
          
          if(mid!=0)
          {
          tots.push({
            Date: mid,
            Quantity: Qty,
          });
          }
        }
    
        res.json(tots);
    
      } catch (error) {
        console.error(error);
        res.status(500).send('Internal server error');
      }
    });



    /// Payments List Function
  app.get('/api/payments', async(req,res) =>{
    try {
      const payments = await db.collection('Payments').find({}).sort({ TransId: -1 }).toArray();
      res.json(payments);
    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });

  /// Last Payment
app.get('/api/payments/last', async(req,res) =>{
  try {
    const LastPayment = await db.collection('Payments').findOne({}, { sort: { TransId: -1 } });
  
    res.json(LastPayment);
  } catch (error) {
    console.error(error);
    res.status(500).send('Internal server error');
  }
  });


    /// Manager Phone
app.get('/api/manager/phone', async(req,res) =>{
  try {
    const Manager = await db.collection('Users').findOne({ accessLevel: "Manager" });;
  
    res.json(Manager);
  } catch (error) {
    console.error(error);
    res.status(500).send('Internal server error');
  }
  });

    /// Insights function
    app.get('/api/insights', async(req,res) =>{
      try {

        let insight;
        
      /// Yestrday and today difference
      /// Yesterday
        const currentDate = new Date();
        let date = currentDate.getDate();;
        let month = currentDate.getMonth(); 
        let year = currentDate.getFullYear();

        const pipeline = [
          {
            $match: {
              Year: year,
              Month: month + 1,
              Date: date - 1
            }
          },
          {
            $group: {
              _id: null,
              totalAccQty: { $sum: "$AccQty" }
            }
          }
        ];
        const results = await db.collection("LeafLots").aggregate(pipeline).toArray();     
        const yesterday = results[0]?.totalAccQty || 0;
       
        /// Today
        const pipeline2 = [
          {
            $match: {
              Year: year,
              Month: month + 1,
              Date: date
            }
          },
          {
            $group: {
              _id: null,
              totalAccQty: { $sum: "$AccQty" }
            }
          }
        ];
        const results1 = await db.collection("LeafLots").aggregate(pipeline2).toArray();     
        const today = results1[0]?.totalAccQty || 0;

        const difference = today -yesterday;


       

        /// New suppliers registration
        const pipeline3 = [
          {
            $match: {
              RegYear: year,
              RegMonth: month + 1,
              
            }
          },
          {
            $group: {
              _id: null,
              totalDocuments: { $sum: 1 }
            }
          }
        ];
        const results3 = await db.collection("Suppliers").aggregate(pipeline3).toArray();     
        const SupCount = results3[0]?.totalDocuments || 0;

        
          /// A grade lots
        const pipeline4 = [
          {
            $match: {
              Year: year,
              Month: month + 1,
              Grade: "a",
              
            }
          },
          {
            $group: {
              _id: null,
              totalDocuments: { $sum: 1 }
            }
          }
        ];
        const results4 = await db.collection("LeafLots").aggregate(pipeline4).toArray();     
        const ACount = results4[0]?.totalDocuments || 0;


        if(difference>0)
          {
            insight=`The factory’s intake increased by ${difference} kilograms compared to yesterday, indicating a steady rise in supply.`
          } else if(SupCount>1)
          {
            insight=`${SupCount} new suppliers have been onboarded this month.`
           }
           else if(ACount>0)
          {
            insight=`The factory recived ${ACount} lots that are in A grade so far this month.`
          } else 
          {
            insight=`No insights available at the moment!`
          }



        res.json(insight);

      } catch (error) {
        console.error(error);
        res.status(500).send('Internal server error');
      }
    });


  /// Lot details Function
  app.get('/api/lot/:id', async(req,res) =>{

    try {
  
      const id = parseInt(req.params.id);
  
      const lot = await db.collection('LeafLots').findOne({ LotId: id });
  
      if (!lot) {
        
        return res.send("LOTNOTFOUND");
      }
  
      return res.json(lot);
      
    } catch (err) {
   
      console.error('Error fetching supplier:', err);
      return res.status(500).json({ message: 'Internal Server Error' });
    }
    
    });



    /// This Month Payment Total
  app.get('/api/payments/thismonth', async(req,res) =>{
    try {

      const currentDate = new Date();
      
      let month = currentDate.getMonth(); 
      let year = currentDate.getFullYear();
      
//pipeline.
const pipeline = [
  {
    
    $match: {
      Month: month+1,
      Year: year
    },
  },
  {
    
    $group: {
      _id: null,
      totalAmount: { $sum: '$Amount' } 
    },
  },
];


const results = await db.collection('Payments').aggregate(pipeline).toArray();
const totPayment = results[0]?.totalAmount || 0;

res.json( totPayment );



    } catch (error) {
      console.error(error);
      res.status(500).send('Internal server error');
    }
  });  


    /// Seesion pending lot to transfer to production
    app.get('/api/lots/pending', async(req,res) =>{
      try {
              
  //pipeline.
  const pipeline = [
    {
      
      $match: {
       Stat:1
      },
    },
    {
      
      $group: {
        _id: null,
        totalAmount: { $sum: '$AccQty' } 
      },
    },
  ];
  
  
  const results = await db.collection('LeafLots').aggregate(pipeline).toArray();
  const Pending = results[0]?.totalAmount || 0;
  
  res.json( Pending );
  
  
  
      } catch (error) {
        console.error(error);
        res.status(500).send('Internal server error');
      }
    });  
    




app.listen(9000, ()=>{

console.log("App is listening on 9000!")

});