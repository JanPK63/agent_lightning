import { math } from '../src/math';

describe('math', () => {

beforeEach(() => {
  jest.clearAllMocks();
});

afterEach(() => {
  jest.restoreAllMocks();
});

  describe('add', () => {
    it('should add two numbers correctly', () => {
      // Arrange
      const a = 2;
      const b = 3;
      // Act
      const result = add(a, b);
      // Assert
      expect(result).toBe(5);
    });

  });

  describe('multiply', () => {
    it('should multiply two numbers', () => {
      // Arrange
      const x = 4;
      const y = 5;
      // Act
      const result = multiply(x, y);
      // Assert
      expect(result).toBe(20);
    });

  });

  describe('divide', () => {
    it('should throw error when dividing by zero', () => {
      // Arrange
      const a = 10;
      const b = 0;
      // Act
      const result = divide(a, b);
      // Assert
      expect(() => result).toThrow();
    });

  });

});