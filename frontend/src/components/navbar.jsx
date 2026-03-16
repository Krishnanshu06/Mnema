import React from "react";
import { Container, Button, Flex, Text, Link, HStack } from "@chakra-ui/react";
// import { IoMoon } from "react-icons/io5";
// import { BiSolidSun } from "react-icons/bi";
const Navbar = () => {
  return (
    <Container
      maxW="container.xl"
      minW={"100vw"}
      px={0}
      margin={0}
      bgColor={"#121919"}
      minH={"8vh"}
    >
      <Flex marginLeft={5} alignItems={"center"} py={2}>
        <Link href="/" _hover={{ textDecoration: "HighlightText" }}>
          <Text fontSize="3xl" fontWeight="bold" color="teal.500" >
            Mnema
          </Text>
        </Link>

        <HStack marginLeft={550}>
          <Button
            colorPalette="teal"
            variant="solid"
            size="lg"
            as={Link}
            href="/"
            borderRadius={15}
          >
            Home
          </Button>
          <Button
            colorPalette="teal"
            variant="solid"
            size="lg"
            as={Link}
            href="/create"
            borderRadius={15}
          >
            Create
          </Button>
        </HStack>
      </Flex>
    </Container>
  );
};

export default Navbar;
