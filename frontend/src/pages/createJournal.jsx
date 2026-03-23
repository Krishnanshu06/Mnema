import {
  Button,
  Container,
  Input,
  Stack,
  Text,
  Textarea,
} from "@chakra-ui/react";
import React from "react";
import { LuCircleArrowRight } from "react-icons/lu";
import { useJournalStore } from "../store/journals";

const CreateJournal = () => {
  
  const [newJournal, setNewJournal] = React.useState({
    title: "",
    content: "",
    userId: "000",
  });

  const { createJournal: CreateJournalApi } = useJournalStore();

  const handleJournalPost = async () => {
    const { success, message } = await CreateJournalApi(newJournal);
    console.log("Success:", success);
    console.log("Message", message);

    setNewJournal({
      title: "",
      content: "",
      userId: "000",
    });
  };

  return (
    <Container
      display={"flex"}
      my={10}
      p={0}
      borderRadius={9}
      bgColor={"#1b3634"}
      maxWidth={"50vw"}
      shadow={"0px 0px 15px 0px #1b3634"}
      textShadow={"0px 0px 10px 0px #ffffff"}
    >
      <Stack direction={"column"} spacing={4} py={5} width={"90%"}>
        <Container alignItems={"center"} justifyContent={"center"}>
          <h1
            style={{
              color: "#a8f3e46b",
              fontSize: "2rem",
              fontWeight: "bold",
            }}
          >
            Create Journal Page
          </h1>
        </Container>

        <Input
          placeholder="Title"
          variant="flushed"
          _placeholder={{ color: "rgba(168, 243, 228, 0.42)" }}
          color={"white"}
          size="lg"
          css={{ "--focus-color": "rgba(168, 243, 228, 0.42)" }}
          marginTop={10}
          mx={10}
          name="title"
          value={newJournal.title}
          onChange={(e) =>
            setNewJournal({ ...newJournal, title: e.target.value })
          }
        />

        <Textarea
          autoresize
          maxH="50vh"
          placeholder="Start writing your journal here..."
          variant="subtle"
          bgColor={"#1b3634"}
          _placeholder={{ color: "rgba(168, 243, 228, 0.42)" }}
          color={"white"}
          size="lg"
          css={{ "--focus-color": "rgba(168, 243, 228, 0.42)" }}
          mx={10}
          marginTop={6}
          name="content"
          value={newJournal.content}
          onChange={(e) =>
            setNewJournal({ ...newJournal, content: e.target.value })
          }
        />

        <Button
          alignSelf={"center"}
          bgColor={"#a8f3e46b"}
          color={"#1b3634"}
          _hover={{
            bgColor: "#2b63576b",
            color: "#cffaf7",
            shadow: "0px 0px 10px 0px #84d3ce",
          }}
          borderRadius={20}
          onClick={handleJournalPost}
        >
          <Text fontSize={18} fontWeight={"bold"}>
            Post
          </Text>
          <LuCircleArrowRight />
        </Button>
      </Stack>
    </Container>
  );
};

export default CreateJournal;
