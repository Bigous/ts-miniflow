import { MSE } from '../src/mse';

test('Should greet with message', () => {
  const greeter = new MSE('friend');
  expect(greeter.greet()).toBe('Bonjour, friend!');
});
