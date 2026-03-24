import React from "react";
import { Container, VStack, Text, SimpleGrid } from "@chakra-ui/react";
import { useJournalStore } from "../store/journals";
import { Link } from "react-router-dom";
import JournalCard from "../components/JournalCard";

const HomePage = () => {

  const { fetchJournals , journals } = useJournalStore();

  React.useEffect(() => {
    fetchJournals();
  }, [fetchJournals]);

  React.useEffect(() => {
    console.log("Journals Updated:", journals);
  }, [journals]);

  return (
    <Container
      display={"flex"}
      my={10}
      p={0}
      borderRadius={9}
      bgColor={"#1b3634"}
      maxWidth={"80%"}
      shadow={"0px 0px 15px 0px #1b3634"}
      textShadow={"0px 0px 10px 0px #ffffff"}
      overflowX={"hidden"}
    >
      <VStack
        marginBottom={20}
        justifyContent={"center"}
        minW={"100%"}
        alignItems={"center"}
      >
        <Text
          fontSize={"2.5rem"}
          fontWeight={"bold"}
          color={"#a8f3e46b"}
          textAlign={"center"}
          m={6}
        >
          Journals
        </Text>

        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={10} w={"90%"}>
          {journals.map((journal) => (
            <JournalCard key={journal._id} journal={journal} />
            
          ))}
        </SimpleGrid>

        {journals.length === 0 && (
          <Text
            fontSize={"1.5rem"}
            fontWeight={"bold"}
            color={"#a8f3e46b"}
            textAlign={"center"}
          >
            No journals{" "}
            <Link to={"/create"}>
              <Text
                as={"span"}
                color={"#a8f3e4d2"}
                _hover={{ textDecoration: "underline" }}
              >
                {" "}
                Create one
              </Text>
            </Link>
          </Text>
        )}
      </VStack>
    </Container>
  );
};

export default HomePage;
