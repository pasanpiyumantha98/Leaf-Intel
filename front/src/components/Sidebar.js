import {
    Drawer,
    List,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Typography,
    Divider,
    Collapse,
    Box,
  } from "@mui/material";
  import {
    Dashboard,
    ShoppingCart,
    ExpandLess,
    ExpandMore,
    BarChart,
    Layers,
  } from "@mui/icons-material";
  import { useState } from "react";
  
  const drawerWidth = 240;
  
  const Sidebar = ({ mobileOpen, onClose, isMobile }) => {
    const [openReports, setOpenReports] = useState(false);
  
    const drawerContent = (
      <Box>
        <Box sx={{ px: 2, py: 2 }}>
          <Typography variant="h6" color="primary">
            Toolpad
          </Typography>
        </Box>
  
        <List>
          <Typography sx={{ pl: 2, pt: 1 }} variant="caption" color="gray">
            Main items
          </Typography>
  
          <ListItemButton>
            <ListItemIcon sx={{ color: "white" }}>
              <Dashboard />
            </ListItemIcon>
            <ListItemText primary="Dashboard" />
          </ListItemButton>
  
          <ListItemButton>
            <ListItemIcon sx={{ color: "white" }}>
              <ShoppingCart />
            </ListItemIcon>
            <ListItemText primary="Orders" />
          </ListItemButton>
  
          <Divider sx={{ my: 1, backgroundColor: "#333" }} />
  
          <Typography sx={{ pl: 2, pt: 1 }} variant="caption" color="gray">
            Analytics
          </Typography>
  
          <ListItemButton onClick={() => setOpenReports(!openReports)}>
            <ListItemIcon sx={{ color: "white" }}>
              <BarChart />
            </ListItemIcon>
            <ListItemText primary="Reports" />
            {openReports ? <ExpandLess /> : <ExpandMore />}
          </ListItemButton>
  
          <Collapse in={openReports} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              <ListItemButton sx={{ pl: 4 }}>
                <ListItemText primary="Monthly Report" />
              </ListItemButton>
              <ListItemButton sx={{ pl: 4 }}>
                <ListItemText primary="Yearly Report" />
              </ListItemButton>
            </List>
          </Collapse>
  
          <ListItemButton>
            <ListItemIcon sx={{ color: "white" }}>
              <Layers />
            </ListItemIcon>
            <ListItemText primary="Integrations" />
          </ListItemButton>
        </List>
      </Box>
    );
  
    return (
      <>
        {/* Mobile Drawer */}
        {isMobile && (
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={onClose}
            ModalProps={{ keepMounted: true }}
            sx={{
              display: { xs: "block", md: "none" },
              "& .MuiDrawer-paper": {
                width: drawerWidth,
                backgroundColor: "#121212",
                color: "#fff",
              },
            }}
          >
            {drawerContent}
          </Drawer>
        )}
  
        {/* Desktop Drawer */}
        {!isMobile && (
          <Drawer
            variant="permanent"
            sx={{
              display: { xs: "none", md: "block" },
              "& .MuiDrawer-paper": {
                width: drawerWidth,
                backgroundColor: "#121212",
                color: "#fff",
              },
            }}
            open
          >
            {drawerContent}
          </Drawer>
        )}
      </>
    );
  };
  
  export default Sidebar;
  