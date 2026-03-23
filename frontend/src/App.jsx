import { Box, Container } from "@chakra-ui/react"
import { Route, Routes } from "react-router-dom"
import HomePage from "./pages/HomePage"
import CreateJournal from "./pages/CreateJournal"
import Navbar from "./components/Navbar"


function App() {

  return (
    <>
    <Container bg={"#222d2d"} margin={0} padding={0} minHeight={"100vh"} minWidth={"100%"}>
      <Navbar />
      <Routes>
        <Route path='/' element= {<HomePage />} />
        <Route path='/home' element= {<HomePage />} />
        <Route path='/create' element= {<CreateJournal />} />
      </Routes>
    </Container>
    </>
  )
}

export default App
