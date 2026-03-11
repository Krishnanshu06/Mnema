import { Button , Box} from "@chakra-ui/react"
import { Route, Routes } from "react-router-dom"
import Home from "./pages/home"
import CreateJournal from "./pages/createJournal"
import Navbar from "./components/navbar"


function App() {

  return (
    <>
    <Box background="tomato" width="100%" padding="4" color="white">
      <Navbar />
      <Routes>
        <Route path='/' element= {<Home />} />
        <Route path='/create' element= {<CreateJournal />} />
      </Routes>
    </Box>
    </>
  )
}

export default App
