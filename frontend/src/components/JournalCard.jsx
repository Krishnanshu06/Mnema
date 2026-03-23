import { useJournalStore } from "../store/journals";
import { Box, HStack, Icon, IconButton, Text } from "@chakra-ui/react";
import React from "react";
// import { DeleteIcon, EditIcon } from "@chakra-ui/icons";
import { CiEdit } from "react-icons/ci";
import { MdDeleteForever } from "react-icons/md";

const JournalCard = ({ journal }) => {

  const {deleteJournals} = useJournalStore();

  const handleDelete = async (jid) => {
    const { success , message } = await deleteJournals(jid)
    console.log("delete  ", success , "message: " ,message)
  }

  return (
    <>
      <Box
        shadow="lg"
        rounded="lg"
        overflow="hidden"
        transition="all 0.3s"
        _hover={{ transform: "translateY(-5px)", shadow: "xl" }}
        m={4}
        bgColor={"#29514e"}
        color={"#a8f3e46b"}
        minW="0"
        p={1}
      >
        <Text
          overflow={"hidden"}
          fontSize={"1.2rem"}
          fontWeight={"bold"}
          p={1}
          paddingLeft={3}
          maxH={9}
          borderRadius={9}
          shadow={"0px 0px 15px 0px "}
        >
          {journal.title}
        </Text>

        <Text
          fontSize={"1rem"}
          // borderRadius={9}
          // shadow={"0px 0px 15px 0px "}
          p={1}
          paddingLeft={3}
          paddingTop={1}
          marginTop={1}
          // paddingBottom={4}
          minH={"14vh"}
          maxH={"20vh"}
          overflow={"hidden"}
        >
          {journal.content}
        </Text>
        <HStack justifyContent={"right"}>
          <IconButton
            aria-label="Edit"
            bg={"#29514e"}
            shadow={"0px 0px 5px 0px #a8f3e46b"}
            rounded="lg"
            color={"#a8f3e46b"}
          _hover={{ bg:"#a8f3e46b"}}
          >
            <CiEdit />
          </IconButton>

          <IconButton
            aria-label="Delete"
            bg={"#29514e"}
            shadow={"0px 0px 5px 0px #a8f3e46b"}
            rounded="lg"
            color={"#a8f3e46b"}
            m={1}
            _hover={{ bg:"#a8f3e46b"}}
            onClick={() => handleDelete(journal._id)}
          >
            <MdDeleteForever />
          </IconButton>
        </HStack>
      </Box>
    </>
  );
};

export default JournalCard;
