import React from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';

const HeaderContainer = styled.header`
  background: linear-gradient(to right, #0a192f, #112240);
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Logo = styled.div`
  color: #64ffda;
  font-size: 1.5rem;
  font-weight: bold;
`;

const Nav = styled.nav`
  display: flex;
  gap: 2rem;
`;

const NavLink = styled(Link)`
  color: white;
  text-decoration: none;
  &:hover {
    color: #64ffda;
  }
`;

const Header = () => {
  return (
    <HeaderContainer>
      <Logo>K2Stats</Logo>
      <Nav>
        <NavLink to="/">Home</NavLink>
        <NavLink to="/prediction">Match Prediction</NavLink>
        <NavLink to="/classement">Classement</NavLink>
        <NavLink to="/about">About Us</NavLink>
      </Nav>
    </HeaderContainer>
  );
};

export default Header;